import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import math


def plot_patches_from_hdf5_single(
    hdf5_path,
    subject_id,
    image_id,
    n_cols=8
):
    """
    Example group names:
        10001_001_L
        10001_001_R
    """

    group_key_base = f"{subject_id}_{image_id}"

    for side in ("L", "R"):

        group_name = f"{group_key_base}_{side}"

        with h5py.File(hdf5_path, 'r') as hf:

            if group_name not in hf:
                print(f"Group '{group_name}' not found.")
                continue

            grp = hf[group_name]

            patches = grp['patches'][:]
            indices = grp['patch_source_point_indices'][:]

        # remove channel dimension
        patches = patches[..., 0]

        n_patches = len(patches)

        if n_patches == 0:
            print(f"No patches in {group_name}")
            continue

        # -------------------------------------------------
        # GRID SIZE
        # -------------------------------------------------

        n_rows = math.ceil(n_patches / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2)
        )

        axes = np.array(axes).flatten()

        # -------------------------------------------------
        # DISPLAY PATCHES
        # -------------------------------------------------

        for idx, patch in enumerate(patches):

            axes[idx].imshow(patch, cmap='gray')

            axes[idx].set_title(
                f"pt {indices[idx]}",
                fontsize=8
            )

            axes[idx].axis('off')

        # hide unused axes
        for idx in range(n_patches, len(axes)):
            axes[idx].axis('off')

        knee_label = "Left (flipped)" if side == "L" else "Right"

        plt.suptitle(
            f"{group_key_base} | {knee_label}",
            fontsize=10
        )

        plt.tight_layout()

        out_path = f"{group_key_base}_{side}_patches.png"

        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved → {out_path}")


if __name__ == "__main__":

    HDF5_PATH = "test_single_file.h5"

    SUBJECT_ID = "10001"
    IMAGE_ID   = "001"

    plot_patches_from_hdf5_single(
        HDF5_PATH,
        SUBJECT_ID,
        IMAGE_ID
    )