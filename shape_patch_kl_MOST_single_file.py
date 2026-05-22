import numpy as np
import re
import math
import cv2
import pydicom
import h5py

# =========================================================
# CONFIG — SINGLE FILE TEST
# =========================================================

DICOM_PATH = "/data/net/datasets/MOST/temp/M0001_PHPP/10001/PA15/100012_clean.dcm"
PTS_PATH = "/data/net/datasets/MOST/temp/M0001_PHPP/10001/PA15/100012_clean.pts"

OUTPUT_HDF5 = "test_single_file.h5"

SUBJECT_ID = "10001"
IMAGE_ID   = "001"

EXPECTED_TOTAL_POINTS = 148
EXPECTED_LANDMARKS_PER_KNEE = 74

IMG_SIZE = 100
CROP_PATCH_SIZE = 16
TARGET_PATCH_SIZE = (CROP_PATCH_SIZE, CROP_PATCH_SIZE)

PATCH_AREA_PX = IMG_SIZE * IMG_SIZE

RANGE1 = np.arange(9, 27)
RANGE2 = np.arange(44, 67)
PATCH_POINT_INDICES = np.concatenate([RANGE1, RANGE2])


# =========================================================
# READ SINGLE .PTS FILE
# =========================================================

def read_pts_file(filepath, expected_points):

    points = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    start_index = end_index = -1

    for i, line in enumerate(lines):

        if line.strip() == "{":
            start_index = i + 1

        elif line.strip() == "}":
            end_index = i
            break

    for line in lines[start_index:end_index]:

        line = line.strip()

        if not line:
            continue

        coords = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)

        if len(coords) == 2:
            points.append([float(coords[0]), float(coords[1])])

    if len(points) != expected_points:
        raise ValueError(
            f"Expected {expected_points} points, got {len(points)}"
        )

    return np.array(points)


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_xray(img, cut_min=5, cut_max=99, multiplier=65535):

    img = img.copy().astype(np.float64)

    lim1, lim2 = np.percentile(img, [cut_min, cut_max])

    img = np.clip(img, lim1, lim2)

    img -= lim1

    img /= img.max() if img.max() != 0 else 1.0

    img *= multiplier

    return img


# =========================================================
# PATCH HELPERS
# =========================================================

def patch_from_point(point, size):

    topLeft  = (int(point[0] - size), int(point[1] - size))
    botRight = (int(point[0] + size), int(point[1] + size))

    return topLeft, botRight


def crop_patch(image, topLeft, botRight):

    x1, y1 = topLeft
    x2, y2 = botRight

    return image[y1:y2, x1:x2]


def create_patches_for_knee(
    processed_image,
    shapes,
    point_indices,
    patch_half_width,
    target_size,
    horizontally_flip=False
):

    patches = []
    successful_indices = []

    for pt_i in point_indices:

        tl, br = patch_from_point(
            shapes[pt_i],
            size=patch_half_width
        )

        raw = crop_patch(processed_image, tl, br)

        if raw is None or raw.size == 0:
            continue

        resized = cv2.resize(
            raw,
            target_size,
            interpolation=cv2.INTER_AREA
        )

        if horizontally_flip:
            resized = np.fliplr(resized)

        patches.append(resized)
        successful_indices.append(pt_i)

    return patches, successful_indices


# =========================================================
# SAVE TO HDF5
# =========================================================

def save_knee_to_hdf5(
    hf,
    group_key,
    knee_side,
    processed_image,
    shapes,
    patch_half_width,
    target_size,
    point_indices,
    flip=False
):

    patches, successful_indices = create_patches_for_knee(
        processed_image,
        shapes,
        point_indices,
        patch_half_width,
        target_size,
        horizontally_flip=flip,
    )

    if patches:
        patches_np = np.expand_dims(
            np.array(patches, dtype=np.float32),
            axis=-1
        )
    else:
        patches_np = np.zeros(
            (0, target_size[0], target_size[1], 1),
            dtype=np.float32
        )

    grp = hf.create_group(group_key)

    grp.create_dataset(
        "patches",
        data=patches_np,
        compression="gzip"
    )

    grp.create_dataset(
        "patch_source_point_indices",
        data=np.array(successful_indices, dtype=np.int32)
    )

    grp.attrs["side"] = knee_side
    grp.attrs["is_flipped"] = bool(flip)

    print(
        f"Saved {patches_np.shape[0]} patches "
        f"→ {group_key}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # LOAD DICOM
    # -----------------------------------------------------

    dcm = pydicom.dcmread(DICOM_PATH)

    image_raw = dcm.pixel_array.astype(np.float64)

    processed = process_xray(image_raw)

    img_h, img_w = image_raw.shape

    print("Image shape:", image_raw.shape)

    # -----------------------------------------------------
    # LOAD 148 LANDMARKS
    # -----------------------------------------------------

    all_points = read_pts_file(
        PTS_PATH,
        EXPECTED_TOTAL_POINTS
    )

    print("All points shape:", all_points.shape)

    # split into 2 knees
    shapes_L = all_points[:74]
    shapes_R = all_points[74:]

    print("Left knee shape :", shapes_L.shape)
    print("Right knee shape:", shapes_R.shape)

    # -----------------------------------------------------
    # PATCH SIZE
    # -----------------------------------------------------

    patch_half_width = (
        math.sqrt(PATCH_AREA_PX / (3560 * 4320))
        * math.sqrt(img_h * img_w)
        / 2
    )

    print("Patch half width:", patch_half_width)

    # -----------------------------------------------------
    # SAVE HDF5
    # -----------------------------------------------------

    with h5py.File(OUTPUT_HDF5, "w") as hf:

        save_knee_to_hdf5(
            hf,
            f"{SUBJECT_ID}_{IMAGE_ID}_L",
            "L",
            processed,
            shapes_L,
            patch_half_width,
            TARGET_PATCH_SIZE,
            PATCH_POINT_INDICES,
            flip=True,
        )

        save_knee_to_hdf5(
            hf,
            f"{SUBJECT_ID}_{IMAGE_ID}_R",
            "R",
            processed,
            shapes_R,
            patch_half_width,
            TARGET_PATCH_SIZE,
            PATCH_POINT_INDICES,
            flip=False,
        )

    print(f"\nDone. Saved to {OUTPUT_HDF5}")


if __name__ == "__main__":
    main()