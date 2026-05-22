import numpy as np
import os
import re
import math
import cv2
import pydicom
import h5py

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# BASE_PTS_DIR   = "/data/net/datasets/MOST/preprocess_Points_Knee_Radiographs"
# BASE_IMG_DIR   = "/data/net/datasets/MOST/preprocess_Knee_Radiographs"
BASE_IMG_DIR = "/data/net/datasets/MOST/image_temp"
BASE_PTS_DIR = "/data/net/datasets/MOST/pts_pixel_temp"
VISIT_FOLDERS = sorted([d for d in os.listdir(BASE_PTS_DIR)
    if os.path.isdir(os.path.join(BASE_PTS_DIR, d))
])  # change per visit, e.g. M0001_PHPP, M0003_PHPP …
VISIT_LABEL    = "M00"          # short label used in output filenames


# Labels – set to None if not yet available; those patients will be skipped
# when kl_grade == -999 inside save_knee_to_hdf5.
# Expected format: CSV / TXT with columns: subject_id, image_id, KL_L, KL_R, …
# Leave as None to collect patches without labels (kl_grade stored as -999).
LABEL_FILE     = "/data/net/datasets/MOST/MOST_KL_labels.csv"            # e.g. "/data/.../most_labels.csv"

IMG_SIZE        = 100           # used for patch-size scaling (match OAI convention) 
CROP_PATCH_SIZE = 16
TARGET_PATCH_SIZE = (CROP_PATCH_SIZE, CROP_PATCH_SIZE)
PATCH_AREA_PX   = IMG_SIZE * IMG_SIZE

EXPECTED_LANDMARKS_PER_KNEE = 74   # each _L.pts / _R.pts has this many points

OUTPUT_NPZ  = f"MOST_{VISIT_LABEL}_shapes_LR.npz"
OUTPUT_HDF5 = f"MOST_{VISIT_LABEL}_knee_patches_{CROP_PATCH_SIZE}_{IMG_SIZE}.h5"

# Landmark index ranges for patch extraction (same as OAI)
RANGE1 = np.arange(9, 27)
RANGE2 = np.arange(44, 67)
PATCH_POINT_INDICES = np.concatenate([RANGE1, RANGE2])


# ─────────────────────────────────────────────
# Landmark / image discovery helpers
# ─────────────────────────────────────────────

def discover_subjects(base_pts_dir, visit_folders):
    """
    Walk:

        base_pts_dir / <visit_folder> / <subject_id> / <PA*> /

    and return records:

        {
            most_id,
            subject_id,
            view_subdir,
            image_id,
            pts_L,
            pts_R
        }
    """

    records = []
    i = 0
    for visit_folder in visit_folders:
        i+=1
        if i >800:
            break
        visit_root = os.path.join(base_pts_dir, visit_folder)

        if not os.path.isdir(visit_root):
            print(f"Warning: visit root not found: {visit_root}")
            continue

        print(f"Scanning {visit_root}")

        for subject_id in sorted(os.listdir(visit_root)):

            subject_dir = os.path.join(visit_root, subject_id)

            if not os.path.isdir(subject_dir):
                continue

            # Find all subdirs starting with "PA"
            pa_dirs = [
                d for d in os.listdir(subject_dir)
                if d.startswith("PA")
                and os.path.isdir(os.path.join(subject_dir, d))
            ]

            for view_subdir in pa_dirs:

                view_dir = os.path.join(
                    subject_dir,
                    view_subdir
                )

                pts_files = os.listdir(view_dir)

                stems = {}

                for fname in pts_files:

                    if fname.endswith("_L.pts"):

                        stem = fname[:-len("_L.pts")]

                        stems.setdefault(stem, {})["L"] = (
                            os.path.join(view_dir, fname)
                        )

                    elif fname.endswith("_R.pts"):

                        stem = fname[:-len("_R.pts")]

                        stems.setdefault(stem, {})["R"] = (
                            os.path.join(view_dir, fname)
                        )

                for stem, sides in stems.items():

                    if "L" not in sides or "R" not in sides:

                        print(
                            f"Warning: incomplete pair for "
                            f"'{stem}' in subject {subject_id}"
                        )

                        continue

                    records.append(
                        dict(
                            most_id=visit_folder,
                            subject_id=subject_id,
                            view_subdir=view_subdir,
                            image_id=stem,
                            pts_L=sides["L"],
                            pts_R=sides["R"],
                        )
                    )

    print(f"Discovered {len(records)} subject/image pairs.")

    return records

def image_path_for(base_img_dir,  visit_folder, subject_id, view_subdir, image_id):
    return os.path.join(base_img_dir, visit_folder, subject_id, view_subdir, image_id+ ".dcm")


# ─────────────────────────────────────────────
# .pts reading  (identical logic to OAI version)
# ─────────────────────────────────────────────

def read_pts_file(filepath, expected_points):
    points = []
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        start_index = end_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "{":
                start_index = i + 1
            elif line.strip() == "}":
                end_index = i
                break

        if start_index == -1 or end_index == -1 or start_index >= end_index:
            print(f"  Warning: no valid {{...}} block in {filepath}")
            return None

        for line in lines[start_index:end_index]:
            line = line.strip()
            if not line:
                continue
            coords = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            if len(coords) == 2:
                points.append([float(coords[0]), float(coords[1])])
            else:
                print(f"  Warning: malformed line in {filepath}: '{line}'")

        if len(points) != expected_points:
            print(
                f"  Warning: expected {expected_points} points, got {len(points)} in {filepath}"
            )
            return None

        return np.array(points)  # (expected_points, 2)

    except FileNotFoundError:
        print(f"  Error: file not found: {filepath}")
        return None
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None


# ─────────────────────────────────────────────
# Label loading  (stub – extend for real labels)
# ─────────────────────────────────────────────
def process_kl(value):
    if value is None or value == "":
        return -999
    
    try:
        return max(0, math.floor(float(value) + 0.5))
    except (ValueError, TypeError):
        return -999

def load_labels(label_file):
    """
    Returns a dict keyed by (subject_id, image_id) →
        { 'kl_L': int, 'kl_R': int, 'aux_L': list, 'aux_R': list }

    Extend this function once you have the MOST label file.
    Currently returns an empty dict so all patients get kl_grade = -999
    and are saved without labels (or skipped, depending on your preference).
    """
    if label_file is None:
        return {}

    import pandas as pd
    df = pd.read_csv(label_file)
    labels = {}

    for _, row in df.iterrows():
        key = str(row["MOSTID"])

        labels[key] = {
            "kl_L": process_kl(row.get("V7XLKL")),
            "kl_R": process_kl(row.get("V7XRKL")),
        }
    return labels


# ─────────────────────────────────────────────
# Image processing  (identical to OAI)
# ─────────────────────────────────────────────

def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    img = img.copy().astype(np.float64)
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img = np.clip(img, lim1, lim2)
    img -= lim1
    img /= img.max() if img.max() != 0 else 1.0
    img *= multiplier
    return img


# ─────────────────────────────────────────────
# Patch helpers  (identical to OAI)
# ─────────────────────────────────────────────

def patch_from_point(point, size):
    topLeft  = (int(point[0] - size), int(point[1] - size))
    botRight = (int(point[0] + size), int(point[1] + size))
    return topLeft, botRight


def crop_patch(image, topLeft, botRight):
    x1, y1 = topLeft
    x2, y2 = botRight
    return image[y1:y2, x1:x2]


def create_patches_for_knee(processed_image, shapes, point_indices,
                            patch_half_width, target_size, horizontally_flip=False):
    patches, successful_indices = [], []
    for pt_i in point_indices:
        if pt_i >= len(shapes):
            continue
        tl, br = patch_from_point(shapes[pt_i], size=patch_half_width)
        raw = crop_patch(processed_image, tl, br)
        if raw is None or raw.size == 0:
            continue
        resized = cv2.resize(raw, target_size, interpolation=cv2.INTER_AREA)
        if horizontally_flip:
            resized = np.fliplr(resized)
        patches.append(resized)
        successful_indices.append(pt_i)
    return patches, successful_indices


# ─────────────────────────────────────────────
# HDF5 saving  (identical to OAI)
# ─────────────────────────────────────────────

def save_knee_to_hdf5(hf, group_key, knee_side, processed_image, shapes,
                      kl_grade, aux_features, patch_half_width,
                      target_size, point_indices, flip=False): #
    """Save patches + metadata for one knee into an open HDF5 file."""
    if kl_grade == -999:
        # Comment out the `return` below if you want to save unlabelled knees too.
        # return
        pass

    patches, successful_indices = create_patches_for_knee(
        processed_image, shapes, point_indices,
        patch_half_width, target_size, horizontally_flip=flip,
    )

    if patches:
        patches_np = np.expand_dims(np.array(patches, dtype=np.float32), axis=-1)
    else:
        patches_np = np.zeros((0, target_size[0], target_size[1], 1), dtype=np.float32)

    grp = hf.create_group(group_key)
    grp.create_dataset("patches",                  data=patches_np,                              compression="gzip")
    grp.create_dataset("kl_grade",                 data=np.array([kl_grade],        dtype=np.int32))
    grp.create_dataset("aux_feature",              data=np.array([aux_features],    dtype=np.int32) if aux_features else np.array([], dtype=np.int32))
    grp.create_dataset("patch_source_point_indices", data=np.array(successful_indices, dtype=np.int32))

    grp.attrs["side"]                    = knee_side
    grp.attrs["is_flipped"]              = bool(flip)
    grp.attrs["target_patch_size"]       = target_size
    grp.attrs["original_num_points"]     = len(shapes)
    grp.attrs["requested_point_indices"] = point_indices.tolist()
    grp.attrs["patch_half_width"]        = patch_half_width

    print(f"    Saved {patches_np.shape[0]} patches → '{group_key}' (KL {kl_grade})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # 1. Discover all subject records
    records = discover_subjects(BASE_PTS_DIR, VISIT_FOLDERS)
    if not records:
        print("No records found. Exiting.")
        return

    # 2. Load labels (empty dict if LABEL_FILE is None)
    labels = load_labels(LABEL_FILE)

    # 3. Iterate records and collect shapes / labels
    most_ids = []
    subject_ids  = []
    image_ids    = []
    view_subdirs = []
    shapes_L_list, shapes_R_list = [], []
    kl_L_list,    kl_R_list     = [], []
    aux_L_list,   aux_R_list    = [], []

    for rec in records:
        most_id = str(rec["most_id"])[:5]
        sid   = rec["subject_id"]
        iid   = rec["image_id"]
        view_subdir = rec["view_subdir"]
        print(f"  Reading landmarks for subject={sid}, image={iid}")

        lm_L = read_pts_file(rec["pts_L"], EXPECTED_LANDMARKS_PER_KNEE)
        lm_R = read_pts_file(rec["pts_R"], EXPECTED_LANDMARKS_PER_KNEE)

        if lm_L is None or lm_R is None:
            print(f"  Skipping {sid} due to landmark errors.")
            continue

        # Labels
        label_entry = labels.get(most_id, {})
        kl_L  = label_entry.get("kl_L",  -999)
        kl_R  = label_entry.get("kl_R",  -999)
        if kl_L == -999 or kl_R == -999:
            print(f"  Skipping {sid} due no labels.")
            continue 

        aux_L = label_entry.get("aux_L", [])
        aux_R = label_entry.get("aux_R", [])

        subject_ids.append(sid)
        image_ids.append(iid)
        most_ids.append(rec["most_id"])
        view_subdirs.append(view_subdir)
        shapes_L_list.append(lm_L)   # already (74, 2)
        shapes_R_list.append(lm_R)
        kl_L_list.append(kl_L)
        kl_R_list.append(kl_R)
        aux_L_list.append(aux_L)
        aux_R_list.append(aux_R)

    print(f"\nCollected {len(subject_ids)} valid records.")

    shapes_L_np  = np.array(shapes_L_list)   # (N, 74, 2)
    shapes_R_np  = np.array(shapes_R_list)
    kl_L_np      = np.array(kl_L_list)
    kl_R_np      = np.array(kl_R_list)

    # 4. Save NPZ
    np.savez(
        OUTPUT_NPZ,
        subject_ids=np.array(subject_ids),
        image_ids=np.array(image_ids),
        shapes_L=shapes_L_np,
        shapes_R=shapes_R_np,
        KL_L=kl_L_np,
        KL_R=kl_R_np,
        aux_L=np.array(aux_L_list, dtype=object),
        aux_R=np.array(aux_R_list, dtype=object),
    )
    print(f"Shapes saved to {OUTPUT_NPZ}")

    # 5. Build HDF5 with patches
    print(f"\nBuilding patch HDF5 → {OUTPUT_HDF5}")
    with h5py.File(OUTPUT_HDF5, "w") as hf:
        dt = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset("subject_ids",   data=np.array(subject_ids, dtype=dt))
        hf.create_dataset("image_ids",     data=np.array(image_ids,   dtype=dt))
        hf.create_dataset("patch_point_indices", data=PATCH_POINT_INDICES)

        for idx, (most_id, sid, iid, view_subdir) in enumerate( zip(most_ids, subject_ids, image_ids, view_subdirs)):
                print(f"  [{idx+1}/{len(subject_ids)}] subject={sid}, image={iid}")
                img_path = image_path_for( BASE_IMG_DIR, most_id, sid, view_subdir, iid)
                try:
                    dcm        = pydicom.dcmread(img_path)
                    image_raw  = dcm.pixel_array.astype(np.float64)
                except FileNotFoundError:
                    print(f"    DICOM not found at {img_path} - skipping.")
                    continue
                except Exception as e:
                    print(f"    DICOM read error: {e} - skipping.")
                    continue

                processed = process_xray(image_raw, 5, 99, 65535)
                img_h, img_w = image_raw.shape

                # Scale patch half-width to image resolution (same formula as OAI)
                # TODO verify this line!!!
                patch_half_width = (math.sqrt(PATCH_AREA_PX / (3560 * 4320)) * math.sqrt(img_h * img_w) / 2)
                # print("image shape", img_h, img_w, patch_half_width)
                # print("pixel spacing:", dcm.PixelSpacing if "PixelSpacing" in dcm else "N/A")
                shapes_L = shapes_L_np[idx]   # (74, 2)
                shapes_R = shapes_R_np[idx]
                kl_L     = int(kl_L_np[idx])
                kl_R     = int(kl_R_np[idx])
                aux_L    = list(aux_L_list[idx])
                aux_R    = list(aux_R_list[idx])

                # Group key: "<subject_id>_<image_id>_L" / "_R"
                key_base = f"{sid}_{iid}"

                save_knee_to_hdf5(
                    hf, f"{key_base}_L", "L",
                    processed, shapes_L, kl_L, aux_L,
                    patch_half_width, TARGET_PATCH_SIZE,
                    PATCH_POINT_INDICES, flip=True,
                )
                save_knee_to_hdf5(
                    hf, f"{key_base}_R", "R",
                    processed, shapes_R, kl_R, aux_R,
                    patch_half_width, TARGET_PATCH_SIZE,
                    PATCH_POINT_INDICES, flip=False,
                )

    print(f"\nDone. HDF5 written to {OUTPUT_HDF5}")


if __name__ == "__main__":
    main()