"""
Face Recognition MCP Server
Exposes tools for face detection, identification, registration, comparison, and clustering.
Compatible with Claude Desktop, Cursor, Antigravity, and any MCP client.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support both MCP v2.x (MCPServer) and MCP v1.x (FastMCP)
try:
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer(
        "face-recognition-mcp",
        dependencies=["numpy", "pillow", "pydantic"],
    )
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP(
        "face-recognition-mcp",
        dependencies=["numpy", "pillow", "pydantic"],
    )

from config import (
    DB_PATH,
    DEFAULT_TOLERANCE,
    PREFERRED_ENGINE,
    SUPPORTED_EXTENSIONS,
)
from engine import (
    BaseFaceEngine,
    DetectedFace,
    FaceBoundingBox,
    MatchCandidate,
    RecognizedFace,
    get_engine,
)
from storage import FaceDatabase

# Initialize storage and engine
db = FaceDatabase(DB_PATH)
_engine: Optional[BaseFaceEngine] = None


def get_active_engine() -> BaseFaceEngine:
    """Lazy-load and cache the face recognition engine."""
    global _engine
    if _engine is None:
        _engine = get_engine(preferred=PREFERRED_ENGINE)
    return _engine


def _validate_image_path(image_path: str) -> Path:
    """Validate that the file exists and is a supported image."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


@mcp.tool()
def get_system_status() -> Dict[str, Any]:
    """
    Check the face recognition server status, active engine, database stats, and configuration.
    """
    engine_name = "None"
    engine_available = False
    try:
        engine = get_active_engine()
        engine_name = engine.name
        engine_available = True
    except Exception as e:
        engine_name = f"Unavailable ({str(e)})"

    return {
        "status": "healthy" if engine_available else "degraded",
        "active_engine": engine_name,
        "database_path": str(DB_PATH),
        "total_face_samples": db.total_count(),
        "registered_people_count": len(db.list_people()),
        "default_tolerance": DEFAULT_TOLERANCE,
    }


@mcp.tool()
def detect_faces(image_path: str) -> Dict[str, Any]:
    """
    Detect all faces present in an image and return their bounding box coordinates.
    
    Args:
        image_path: Absolute or relative path to the image file.
    """
    path = _validate_image_path(image_path)
    engine = get_active_engine()
    detected = engine.detect_faces(str(path))

    return {
        "image_path": str(path),
        "faces_detected": len(detected),
        "faces": [face.to_dict() for face in detected],
    }


@mcp.tool()
def register_face(name: str, image_path: str, notes: str = "") -> Dict[str, Any]:
    """
    Register a known person into the face database using a reference image.
    The image should contain a clear view of the person's face.
    
    Args:
        name: Name of the individual (e.g. 'Alice', 'John Doe').
        image_path: Path to the reference photo.
        notes: Optional context or tags (e.g. 'Passport photo', 'Event 2026').
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Person name cannot be blank.")

    path = _validate_image_path(image_path)
    engine = get_active_engine()
    encodings_data = engine.extract_encodings(str(path))

    if not encodings_data:
        return {
            "success": False,
            "error": f"No face could be detected in '{image_path}'. Please provide an image with a clear face.",
        }

    # Use primary face (first detected)
    face_meta, primary_encoding = encodings_data[0]
    face_id = db.add_face(
        name=clean_name,
        encoding=primary_encoding,
        source_image=str(path),
        notes=notes,
    )

    warning = None
    if len(encodings_data) > 1:
        warning = (
            f"Image contained {len(encodings_data)} faces. "
            f"Registered the primary face located at {face_meta.box.to_dict()}."
        )

    return {
        "success": True,
        "face_id": face_id,
        "name": clean_name,
        "box": face_meta.box.to_dict(),
        "warning": warning,
        "total_samples_for_person": len(
            [p for p in db.list_people() if p["name"].lower() == clean_name.lower()]
        ),
    }


@mcp.tool()
def match_faces(
    image_path: str,
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Identify and recognize all people present in an image by matching them against the registered face database.
    
    Args:
        image_path: Path to the image file to analyze.
        tolerance: Distance threshold for matching. Lower is stricter (e.g. 0.5), higher is looser (e.g. 0.65). Default is 0.6.
    """
    path = _validate_image_path(image_path)
    engine = get_active_engine()
    threshold = tolerance if tolerance is not None else DEFAULT_TOLERANCE

    known_records = db.get_all_faces()
    if not known_records:
        # Detect faces anyway and report that no known people are registered yet
        detected = engine.detect_faces(str(path))
        return {
            "image_path": str(path),
            "faces_found": len(detected),
            "identified": [],
            "unknown": len(detected),
            "matches": [
                {
                    "box": f.box.to_dict(),
                    "identified_name": "Unknown",
                    "reason": "Face database is currently empty. Use register_face first.",
                }
                for f in detected
            ],
        }

    encodings_data = engine.extract_encodings(str(path))
    results: List[Dict[str, Any]] = []
    identified_names: List[str] = []
    unknown_count = 0

    for face_meta, query_encoding in encodings_data:
        candidates: List[MatchCandidate] = []

        for record in known_records:
            dist = engine.compute_distance(query_encoding, record["encoding"])
            # Approximate percentage similarity from distance
            sim = max(0.0, min(100.0, (1.0 - (dist / 1.0)) * 100.0))
            is_match = dist <= threshold
            candidates.append(
                MatchCandidate(
                    name=record["name"],
                    distance=dist,
                    similarity_percentage=sim,
                    is_match=is_match,
                )
            )

        # Sort candidate matches by lowest distance
        candidates.sort(key=lambda c: c.distance)

        best_match = candidates[0] if candidates and candidates[0].is_match else None
        if best_match:
            identified_name = best_match.name
            if identified_name not in identified_names:
                identified_names.append(identified_name)
            recognized = RecognizedFace(
                box=face_meta.box,
                identified_name=identified_name,
                confidence=face_meta.confidence,
                distance=best_match.distance,
                similarity_percentage=best_match.similarity_percentage,
                top_candidates=candidates[:3],
            )
        else:
            unknown_count += 1
            recognized = RecognizedFace(
                box=face_meta.box,
                identified_name="Unknown",
                confidence=face_meta.confidence,
                distance=candidates[0].distance if candidates else None,
                similarity_percentage=candidates[0].similarity_percentage if candidates else None,
                top_candidates=candidates[:3],
            )

        results.append(recognized.to_dict())

    return {
        "image_path": str(path),
        "faces_found": len(encodings_data),
        "identified": identified_names,
        "unknown_count": unknown_count,
        "tolerance_used": threshold,
        "faces": results,
    }


@mcp.tool()
def compare_two_faces(
    image_path_1: str,
    image_path_2: str,
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare the faces in two images to verify if they are the same person.
    
    Args:
        image_path_1: Path to the first image.
        image_path_2: Path to the second image.
        tolerance: Distance threshold. Default 0.6.
    """
    path1 = _validate_image_path(image_path_1)
    path2 = _validate_image_path(image_path_2)
    engine = get_active_engine()
    threshold = tolerance if tolerance is not None else DEFAULT_TOLERANCE

    enc1 = engine.extract_encodings(str(path1))
    if not enc1:
        raise ValueError(f"No face detected in '{image_path_1}'")

    enc2 = engine.extract_encodings(str(path2))
    if not enc2:
        raise ValueError(f"No face detected in '{image_path_2}'")

    face1_meta, vec1 = enc1[0]
    face2_meta, vec2 = enc2[0]

    distance = engine.compute_distance(vec1, vec2)
    is_same_person = distance <= threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance / 1.0)) * 100.0))

    return {
        "is_same_person": bool(is_same_person),
        "distance": round(distance, 4),
        "similarity_percentage": round(similarity, 2),
        "tolerance": threshold,
        "face_1_box": face1_meta.box.to_dict(),
        "face_2_box": face2_meta.box.to_dict(),
    }


@mcp.tool()
def cluster_faces(
    image_paths: List[str],
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Group a collection of images by facial identity (useful for organizing photo albums).
    
    Args:
        image_paths: List of file paths to images to cluster.
        tolerance: Matching threshold. Default 0.6.
    """
    engine = get_active_engine()
    threshold = tolerance if tolerance is not None else DEFAULT_TOLERANCE

    valid_paths = [_validate_image_path(p) for p in image_paths]
    all_face_records: List[Dict[str, Any]] = []

    for path in valid_paths:
        try:
            encs = engine.extract_encodings(str(path))
            for face_meta, vec in encs:
                all_face_records.append(
                    {
                        "image_path": str(path),
                        "box": face_meta.box.to_dict(),
                        "encoding": vec,
                    }
                )
        except Exception:
            continue

    # Simple greedy community clustering
    clusters: List[Dict[str, Any]] = []
    assigned = [False] * len(all_face_records)

    for i, item in enumerate(all_face_records):
        if assigned[i]:
            continue

        cluster_items = [
            {"image_path": item["image_path"], "box": item["box"]}
        ]
        assigned[i] = True

        for j in range(i + 1, len(all_face_records)):
            if assigned[j]:
                continue
            dist = engine.compute_distance(item["encoding"], all_face_records[j]["encoding"])
            if dist <= threshold:
                assigned[j] = True
                cluster_items.append(
                    {
                        "image_path": all_face_records[j]["image_path"],
                        "box": all_face_records[j]["box"],
                    }
                )

        clusters.append(
            {
                "cluster_id": len(clusters) + 1,
                "face_count": len(cluster_items),
                "images": list(dict.fromkeys([ci["image_path"] for ci in cluster_items])),
                "faces": cluster_items,
            }
        )

    return {
        "total_images_processed": len(valid_paths),
        "total_faces_found": len(all_face_records),
        "clusters_count": len(clusters),
        "clusters": clusters,
    }


@mcp.tool()
def list_known_people() -> Dict[str, Any]:
    """
    List all people registered in the face database along with their sample counts and timestamps.
    """
    people = db.list_people()
    return {
        "total_people": len(people),
        "total_samples": db.total_count(),
        "people": people,
    }


@mcp.tool()
def delete_known_person(name: str) -> Dict[str, Any]:
    """
    Remove a person and all of their face samples from the database.
    
    Args:
        name: Name of the person to remove.
    """
    clean_name = name.strip()
    deleted_count = db.delete_person(clean_name)
    return {
        "success": deleted_count > 0,
        "name": clean_name,
        "records_deleted": deleted_count,
    }


def main():
    """Run the FastMCP server via stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
