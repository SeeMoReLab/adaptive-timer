#!/usr/bin/env python3
"""Plot a CppBedrock PBFT run (see plot_bedrock.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bedrock import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("pbft"))
