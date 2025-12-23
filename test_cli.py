#!/usr/bin/env python3
"""Test foldism CLI with Insulin sequence."""

import subprocess
from pathlib import Path

INSULIN_FASTA = """>Insulin
GIVEQCCTSICSLYQLENYCN
>InsulinB
FVNQHLCGSHLVEALYLVCGERGFFYTPKT
"""

if __name__ == "__main__":
    test_path = Path("test.faa")
    test_path.write_text(INSULIN_FASTA)
    print(f"Created {test_path}")

    cmd = ["uv", "run", "modal", "run", "foldism.py", "--input-faa", str(test_path), "--algorithms", "chai1"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
