"""Compatibility launcher; historical source moved to benchmarks/legacy."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "benchmarks" / "legacy" / "visualize.py"
    print(f"Historical entry point: {target}; see benchmarks/legacy/README.md", file=sys.stderr)
    runpy.run_path(str(target), run_name="__main__")
