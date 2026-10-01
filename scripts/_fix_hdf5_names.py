#!/usr/bin/env python3
"""Fix leftover hdf5_saver/hdf5_loader names after signature migration."""
from pathlib import Path
import re

ROOT = Path("/home/johannes/work/tenpy/cyten_main")

for path in list((ROOT / "src").rglob("*.cpp")) + list((ROOT / "include").rglob("*.h")):
    text = path.read_text()
    if "hdf5_saver" not in text and "hdf5_loader" not in text:
        continue
    if "save_hdf5" not in text and "from_hdf5" not in text and "load_hdf5" not in text:
        continue
    new = text.replace("hdf5_saver", "saver").replace("hdf5_loader", "loader")
    # Avoid double-renames of already-correct names: saver was already saver
    # Fix accidental Saver& saver -> already fine
    # Fix get_attr static calls: loader.attr("get_attr") -> need manual later
    if new != text:
        path.write_text(new)
        print(path.relative_to(ROOT))
