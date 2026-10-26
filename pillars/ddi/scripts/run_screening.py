"""Pillar-specific entry point; implementation remains in the shared project code."""
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[3]
TARGET = REPO / r"src/training/run_experiment_suite.py"
if not TARGET.is_file():
    raise FileNotFoundError(f"Shared implementation not found: {TARGET}")
for path in (REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
