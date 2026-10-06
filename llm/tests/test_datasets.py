"""Rebuild every data set from its seed and check it is byte-identical to data/ (about a minute)."""
import subprocess
import sys
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]


def test_datasets_rebuild_identically(tmp_path):
    result = subprocess.run([sys.executable, str(LLM_DIR / "scripts/build_datasets.py"), "--out", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all identical" in result.stdout
