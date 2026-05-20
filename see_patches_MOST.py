import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


def plot_patches_from_hdf5_most(hdf5_path, subject_id, image_id, side,
                                 n_cols=8, save_path=None):
    """
    subject_id : e.g. "10002A"
    image_id   : e.g. "10002A_00_PA10_L_KNEE" (the stem from the .pts filename)
    side       : "L" or "R"
    """
    with h5py.File(hdf5_path, 'r') as hf:
        group_name = f"{subject_id}_{image_id}_{side}"
        if group_name not in hf:
            print(f"Group '{group_name}' not found. Available keys (first 10):")
            print(list(hf.keys())[:10])
            return

        grp = hf[group_name]
        patches = grp['patches'][:]                        # (N, H, W, 1)
        kl      = grp['kl_grade'][0]
        indices = grp['patch_source_point_indices'][:]
        aux     = grp['aux_feature'][:] if 'aux_feature' in grp and grp['aux_feature'].size > 0 else []

    patches = patches[..., 0]                             # (N, H, W)
    n_patches = len(patches)
    if n_patches == 0:
        print(f"No patches found for {group_name}.")
        return

    n_rows = math.ceil(n_patches / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).flatten()

    for idx, patch in enumerate(patches):
        axes[idx].imshow(patch, cmap='gray')
        axes[idx].set_title(f"pt {indices[idx]}", fontsize=8)
        axes[idx].axis('off')

    for idx in range(n_patches, len(axes)):
        axes[idx].axis('off')

    knee_label = "Left (flipped)" if side == "L" else "Right"
    plt.suptitle(
        f"Subject {subject_id} | Image {image_id} | {knee_label} | KL={kl}"
        + (f" | Aux={list(aux)}" if len(aux) > 0 else ""),
        fontsize=10,
    )
    plt.tight_layout()

    out = save_path or f"{subject_id}_{image_id}_{side}_patches.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out}")


def list_subjects(hdf5_path, n=20):
    """Print the first n group keys to help find valid subject/image_id combos."""
    with h5py.File(hdf5_path, 'r') as hf:
        keys = [k for k in hf.keys() if k not in ("subject_ids", "image_ids", "patch_point_indices")]
        print(f"Total patch groups: {len(keys)}")
        print("First entries:")
        for k in keys[:n]:
            print(" ", k)


if __name__ == "__main__":
    OUTPUT_HDF5 = "MOST_M00_knee_patches_16_100.h5"

    # Step 1: inspect what's inside
    list_subjects(OUTPUT_HDF5)

    # Step 2: pick a subject and plot
    # Copy a key from list_subjects output, e.g. "10002A_10002A_00_PA10_L_KNEE_L"
    # and split it accordingly:
    subject_id = "10002A"
    image_id   = "10002A_00_PA10_L_KNEE"   # the stem between subject_id_ and _L/_R

    plot_patches_from_hdf5_most(OUTPUT_HDF5, subject_id, image_id, "L",
                                 save_path="most_patches_L.png")
    plot_patches_from_hdf5_most(OUTPUT_HDF5, subject_id, image_id, "R",
                                 save_path="most_patches_R.png")