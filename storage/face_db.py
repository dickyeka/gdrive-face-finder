import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FaceDatabase:
    """Manages persistent SQLite storage of facial embeddings and metadata."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding_json TEXT NOT NULL,
                    source_image TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_known_faces_name ON known_faces(name);
                """
            )
            conn.commit()

    def add_face(
        self,
        name: str,
        encoding: List[float],
        source_image: Optional[str] = None,
        notes: str = "",
    ) -> int:
        """Store a new face vector for a given person."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Person name cannot be empty.")
        if not encoding:
            raise ValueError("Encoding vector cannot be empty.")

        now_iso = datetime.now(timezone.utc).isoformat()
        encoding_str = json.dumps(encoding)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO known_faces (name, encoding_json, source_image, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_name, encoding_str, source_image or "", notes or "", now_iso),
            )
            conn.commit()
            return cursor.lastrowid or -1

    def get_all_faces(self) -> List[Dict[str, Any]]:
        """Retrieve all stored faces with deserialized encodings."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, encoding_json, source_image, notes, created_at
                FROM known_faces
                ORDER BY name ASC, created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "encoding": json.loads(row["encoding_json"]),
                    "source_image": row["source_image"],
                    "notes": row["notes"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def list_people(self) -> List[Dict[str, Any]]:
        """Return a aggregated summary of known individuals and sample counts."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    name, 
                    COUNT(id) as samples_count, 
                    MAX(created_at) as last_updated,
                    GROUP_CONCAT(source_image, ' | ') as source_images
                FROM known_faces
                GROUP BY name
                ORDER BY name ASC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "name": row["name"],
                    "samples_count": row["samples_count"],
                    "last_updated": row["last_updated"],
                    "source_images": [
                        img.strip()
                        for img in (row["source_images"] or "").split(" | ")
                        if img.strip()
                    ],
                }
                for row in rows
            ]

    def delete_person(self, name: str) -> int:
        """Remove all face samples associated with a person name."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM known_faces WHERE LOWER(name) = LOWER(?)", (name.strip(),)
            )
            conn.commit()
            return cursor.rowcount

    def delete_face_by_id(self, face_id: int) -> bool:
        """Delete an individual face encoding by database ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM known_faces WHERE id = ?", (face_id,))
            conn.commit()
            return cursor.rowcount > 0

    def total_count(self) -> int:
        """Return total count of stored face vectors."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM known_faces")
            return cursor.fetchone()[0]
