import h5py
import numpy as np

# ─── 1. NPZ FILE ───────────────────────────────────────────
npz = np.load("MOST_M00_shapes_LR.npz", allow_pickle=True)

print("=" * 50)
print("NPZ FILE: MOST_M00_shapes_LR.npz")
print("=" * 50)
print(f"Keys: {list(npz.keys())}")
for key in npz.keys():
    arr = npz[key]
    print(f"\n  [{key}]  shape={arr.shape}  dtype={arr.dtype}")
    print(f"   First 10 values: {arr.flat[:10]}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"   Min={arr.min()}  Max={arr.max()}  Unique={np.unique(arr)[:20]}")

# ─── 2. H5 FILE ────────────────────────────────────────────
print("\n" + "=" * 50)
print("H5 FILE: MOST_M00_knee_patches_16_100.h5")
print("=" * 50)

with h5py.File("MOST_M00_knee_patches_16_100.h5", "r") as hf:

    print(f"Top-level keys ({len(hf.keys())}): {list(hf.keys())[:10]} ...")

    # Print structure of first 3 groups
    top_keys = list(hf.keys())
    sample_keys = [k for k in top_keys if k not in {"subject_ids", "image_ids", "patch_point_indices"}]

    print(f"\n--- Metadata datasets ---")
    for meta_key in ["subject_ids", "image_ids", "patch_point_indices"]:
        if meta_key in hf:
            ds = hf[meta_key]
            print(f"  [{meta_key}]  shape={ds.shape}  dtype={ds.dtype}")
            print(f"   First 5: {ds[:5]}")

    print(f"\n--- First 3 sample groups ---")
    for key in sample_keys[:3]:
        item = hf[key]
        print(f"\n  Group/Dataset: [{key}]  type={type(item)}")
        if isinstance(item, h5py.Group):
            for subkey in item.keys():
                ds = item[subkey]
                print(f"    [{subkey}]  shape={ds.shape}  dtype={ds.dtype}  val={ds[()].flat[:3]}")
        elif isinstance(item, h5py.Dataset):
            print(f"    shape={item.shape}  dtype={item.dtype}")
            print(f"    value={item[()]}")
#----------------------    
#---- checks classes
# import h5py
# import numpy as np
# from collections import Counter

# with h5py.File("MOST_M00_knee_patches_16_100.h5", "r") as hf:

#     excluded = {"subject_ids", "image_ids", "patch_point_indices"}
#     all_keys = [k for k in hf.keys() if k not in excluded]

#     all_grades = []
#     for key in all_keys:
#         grade = int(hf[key]["kl_grade"][0])
#         all_grades.append(grade)

# all_grades = np.array(all_grades)
# counts = Counter(all_grades)

# print(f"Total samples     : {len(all_grades)}")
# print(f"Unique KL grades  : {sorted(counts.keys())}")
# print(f"\nPer-class counts:")
# for grade in sorted(counts.keys()):
#     print(f"  KL {grade} : {counts[grade]} samples")

# print(f"\nExpected classes 0-4 present:")
# for c in range(5):
#     status = "✅" if c in counts else "❌ MISSING"
#     print(f"  Class {c}: {status}")

# print(f"\nUnexpected grades (outside 0-4):")
# unexpected = [g for g in counts if g not in range(5)]
# if unexpected:
#     for g in unexpected:
#         print(f"  KL {g}: {counts[g]} samples  ← PROBLEM")
# else:
#     print("  None")