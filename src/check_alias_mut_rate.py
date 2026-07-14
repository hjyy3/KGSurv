"""Check mutation rates of alias genes across cohorts."""
import glob
import pandas as pd
from pathlib import Path

ALIAS_GENES = ['FAM46C','PAK7','MRE11A','PARK2','WHSC1','SETD8']

print("=== Mutation rates for alias genes across cohorts ===")
for f in sorted(glob.glob("output/processed/valid_*_mut.csv")):
    name = Path(f).stem.replace("valid_", "").replace("_mut", "")
    df = pd.read_csv(f, index_col=0)
    found = {g: df[g].mean() * 100 for g in ALIAS_GENES if g in df.columns}
    if found:
        rate_str = "  ".join(f"{g}={v:.1f}%" for g, v in found.items() if v > 0)
        if rate_str:
            print(f"  {name}: {rate_str}")
        else:
            print(f"  {name}: all 0%")
