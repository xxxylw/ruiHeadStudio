import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_reference_module():
    module_path = Path(__file__).resolve().parents[1] / "threestudio" / "utils" / "reference_sheet.py"
    spec = importlib.util.spec_from_file_location("reference_sheet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_reference_sheet = load_reference_module().load_reference_sheet


class TestReferenceSheet(unittest.TestCase):
    def test_load_reference_sheet_resolves_images_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "front.png").write_bytes(b"fake-png")
            metadata = {
                "identity_mode": "fictional",
                "prompt": "a weathered architect with a charcoal jacket",
                "references": [
                    {
                        "image": "front.png",
                        "view": "front",
                        "weight": 1.0,
                        "face_crop": [12, 8, 52, 52],
                        "person_crop": [4, 2, 60, 64],
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            sheet = load_reference_sheet(root / "metadata.json")

            self.assertEqual(sheet.identity_mode, "fictional")
            self.assertEqual(sheet.prompt, "a weathered architect with a charcoal jacket")
            self.assertEqual(len(sheet.references), 1)
            self.assertEqual(sheet.references[0].image_path, root / "front.png")
            self.assertEqual(sheet.references[0].view, "front")
            self.assertEqual(sheet.references[0].face_crop, (12, 8, 52, 52))
            self.assertEqual(sheet.references[0].person_crop, (4, 2, 60, 64))

    def test_load_reference_sheet_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "identity_mode": "fictional",
                "prompt": "a coherent generated character",
                "references": [
                    {
                        "image": "missing.png",
                        "view": "front",
                        "face_crop": [10, 10, 40, 40],
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_reference_sheet(root / "metadata.json")

    def test_validate_reference_sheet_cli_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "front.png").write_bytes(b"fake-png")
            metadata = {
                "identity_mode": "target_person",
                "prompt": "Cristiano Ronaldo generated reference",
                "references": [
                    {
                        "image": "front.png",
                        "view": "front",
                        "face_crop": [10, 10, 40, 40],
                    }
                ],
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "scripts/validate_reference_sheet.py", str(metadata_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("identity_mode: target_person", result.stdout)
            self.assertIn("references: 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
