"""Regression checks: optimization must not disable provenance gates.

Run: python -m unittest discover -s scripts -p test_validation.py
Report tests require the report's default Windows fonts.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "target" / "validation-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def write(self, name, value):
        (self.output / name).write_text(json.dumps(value), encoding="utf-8")

    def invoke(self, script):
        return subprocess.run(
            [sys.executable, "-O", str(ROOT / "scripts" / script), str(self.output)],
            capture_output=True, text=True, timeout=30,
        )

    def test_audit_rejects_stale_source_under_optimization(self):
        self.write("run.json", {"status": "complete", "source_fingerprint": "stale"})
        result = self.invoke("audit.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rebuild and rerun", result.stderr)
        self.assertFalse((self.output / "evidence_manifest.json").exists())

    @unittest.skipUnless(Path("C:/Windows/Fonts/msjh.ttc").exists(), "Default report fonts require Windows")
    def test_report_rejects_changed_artifact_under_optimization(self):
        run = {"status": "complete"}
        self.write("run.json", run)
        self.write("evidence_manifest.json", {
            "status": "passed", "run": run,
            "artifact_sha256": {"changed.csv": hashlib.sha256(b"original").hexdigest()},
            "source_sha256": {},
        })
        (self.output / "changed.csv").write_bytes(b"modified")
        result = self.invoke("build_report.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Changed artifact: changed.csv", result.stderr)
        self.assertFalse((self.output / "pdf").exists())

    @unittest.skipUnless(Path("C:/Windows/Fonts/msjh.ttc").exists(), "Default report fonts require Windows")
    def test_report_rejects_changed_source_under_optimization(self):
        run = {"status": "complete"}
        self.write("run.json", run)
        self.write("evidence_manifest.json", {
            "status": "passed", "run": run,
            "artifact_sha256": {}, "source_sha256": {"Cargo.toml": "0" * 64},
        })
        result = self.invoke("build_report.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Changed source: Cargo.toml", result.stderr)
        self.assertFalse((self.output / "pdf").exists())


if __name__ == "__main__":
    unittest.main()
