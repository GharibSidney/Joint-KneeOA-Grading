import h5py
import numpy as np
import pandas as pd
from collections import Counter

# ── Load H5 keys and grades ──────────────────────────────
with h5py.File("MOST_M00_knee_patches_16_100.h5", "r") as hf:
    excluded = {"subject_ids", "image_ids", "patch_point_indices"}
    all_keys = [k for k in hf.keys() if k not in excluded]
    h5_grades = {k: int(hf[k]["kl_grade"][0]) for k in all_keys}

# ── Load your CSV ────────────────────────────────────────
df = pd.read_csv("/data/net/datasets/MOST/MOST_labels.csv")   # ← change filename
print("CSV columns:", df.columns.tolist())
print("CSV shape  :", df.shape)
print(df.head())

# ── Extract IDs from H5 keys ─────────────────────────────
# H5 keys look like: '10001_100012_clean_L'
h5_ids = set(all_keys)
csv_ids = set(df["MOSTID"])   # ← change column name

print(f"\nH5  samples          : {len(h5_ids)}")
print(f"CSV samples          : {len(csv_ids)}")
print(f"In H5 but NOT in CSV : {len(h5_ids - csv_ids)}")
print(f"In CSV but NOT in H5 : {len(csv_ids - h5_ids)}")
print(f"In both              : {len(h5_ids & csv_ids)}")

# ── Check KL grade distribution for H5-only samples ──────
h5_only = h5_ids - csv_ids
h5_only_grades = Counter(h5_grades[k] for k in h5_only)
print(f"\nKL grade distribution of H5-only samples (not in CSV):")
for grade in sorted(h5_only_grades):
    print(f"  KL {grade} : {h5_only_grades[grade]}")