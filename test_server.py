"""
Unit and integration test suite for face-recognition-mcp.
Tests SQLite storage and logic without requiring live camera/images.
"""

import tempfile
import unittest
from pathlib import Path

from storage import FaceDatabase
from engine.base import FaceBoundingBox, DetectedFace


class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_faces.sqlite3"
        self.db = FaceDatabase(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_add_and_list_faces(self):
        # Insert 2 samples for Alice, 1 for Bob
        sample_vec_alice = [0.1] * 128
        sample_vec_bob = [0.9] * 128

        id1 = self.db.add_face("Alice", sample_vec_alice, source_image="/tmp/alice1.jpg")
        id2 = self.db.add_face("Alice", sample_vec_alice, source_image="/tmp/alice2.jpg")
        id3 = self.db.add_face("Bob", sample_vec_bob, source_image="/tmp/bob1.jpg")

        self.assertGreater(id1, 0)
        self.assertGreater(id2, 0)
        self.assertGreater(id3, 0)

        self.assertEqual(self.db.total_count(), 3)

        people = self.db.list_people()
        self.assertEqual(len(people), 2)

        alice_entry = next(p for p in people if p["name"] == "Alice")
        self.assertEqual(alice_entry["samples_count"], 2)
        self.assertEqual(len(alice_entry["source_images"]), 2)

        bob_entry = next(p for p in people if p["name"] == "Bob")
        self.assertEqual(bob_entry["samples_count"], 1)

    def test_delete_person(self):
        self.db.add_face("Charlie", [0.5] * 128)
        self.assertEqual(self.db.total_count(), 1)

        deleted = self.db.delete_person("Charlie")
        self.assertEqual(deleted, 1)
        self.assertEqual(self.db.total_count(), 0)

    def test_face_bounding_box(self):
        box = FaceBoundingBox(top=10, right=100, bottom=90, left=20)
        self.assertEqual(box.x, 20)
        self.assertEqual(box.y, 10)
        self.assertEqual(box.width, 80)
        self.assertEqual(box.height, 80)
        d = box.to_dict()
        self.assertEqual(d["width"], 80)


if __name__ == "__main__":
    unittest.main()
