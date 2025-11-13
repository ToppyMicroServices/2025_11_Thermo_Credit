import os, sys
from pathlib import Path
from typing import List

# Reuse the per-region compute function
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.regions import compute_region  # type: ignore

REGIONS: List[str] = ["jp", "eu", "us"]

def main():
    # Ensure site directory exists
    Path("site").mkdir(parents=True, exist_ok=True)
    out_files = []
    for r in REGIONS:
        try:
            out = compute_region(r)
            out_files.append(out)
        except Exception as e:
            print(f"[warn] Region {r} failed: {e}")
    print("Wrote:", ", ".join(out_files))

if __name__ == "__main__":
    main()
