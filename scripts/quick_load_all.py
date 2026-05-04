"""Fast smoke: load every CSV under data/ and print shape + columns (first file only)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data"
paths = sorted(data_dir.rglob("*.csv"))
if not paths:
    raise SystemExit(f"No CSVs under {data_dir}")

for p in paths:
    df = pd.read_csv(p)
    rel = p.relative_to(ROOT)
    print(f"{rel}: {df.shape[0]:,} rows × {df.shape[1]} cols")

print(f"\nLoaded {len(paths)} CSV files OK.")
