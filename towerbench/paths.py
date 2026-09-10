from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TOWERBENCH_ROOT", Path.home() / "towerbench"))
RAW = ROOT / "data" / "raw"
PREPARED = ROOT / "data" / "prepared"
RESULTS = ROOT / "results"
for _p in (RAW, PREPARED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
