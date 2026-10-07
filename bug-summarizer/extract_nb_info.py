"""Extract training args source code from cell 8 and dataset stats."""
import json, sys, csv
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

# Get training args from notebook
nb = json.load(open("notebook/train_codebert.ipynb", "r", encoding="utf-8"))
cell8_src = "".join(nb["cells"][8].get("source", []))
print("=== CELL 8 SOURCE (Training) ===")
print(cell8_src)

# Get cell 13 (limitations) 
cell13_src = "".join(nb["cells"][13].get("source", []))
print("\n=== CELL 13 SOURCE (Limitations) ===")
print(cell13_src)

# Dataset stats
print("\n=== DATASET STATS ===")
import pandas as pd
df = pd.read_csv("data/bug_dataset.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nSplit distribution:")
print(df['split'].value_counts())
print(f"\nClass distribution:")
print(df['bug_type'].value_counts())
print(f"\nSplit x Class:")
for split in ['train', 'val', 'test']:
    sdf = df[df['split'] == split]
    print(f"\n{split}: {len(sdf)} samples")
    print(sdf['bug_type'].value_counts())
