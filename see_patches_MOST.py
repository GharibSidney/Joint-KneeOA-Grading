import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


def plot_patches_from_hdf5_most(hdf5_path, group_key_base, n_cols=8, save_path_prefix=None):
    """
    group_key_base : everything before the final _L / _R
                     e.g. "10001_100012_clean.dcm"
    """
    for side in ("L", "R"):
        group_name = f"{group_key_base}_{side}"

        with h5py.File(hdf5_path, 'r') as hf:
            if group_name not in hf:
                print(f"Group '{group_name}' not found — skipping.")
                continue

            grp     = hf[group_name]
            patches = grp['patches'][:]
            kl      = grp['kl_grade'][0]
            indices = grp['patch_source_point_indices'][:]
            aux     = grp['aux_feature'][:] if grp['aux_feature'].size > 0 else []

        patches   = patches[..., 0]
        n_patches = len(patches)
        if n_patches == 0:
            print(f"No patches in {group_name}.")
            continue

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
        plt.suptitle(f"{group_key_base} | {knee_label} | KL={kl}", fontsize=10)
        plt.tight_layout()

        prefix = save_path_prefix or group_key_base.replace(".", "_")
        out = f"{prefix}_{side}_patches.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {out}")


if __name__ == "__main__":
    OUTPUT_HDF5 = "MOST_M00_knee_patches_16_100.h5"

    # Plot first 2 patients
    patients = [
        "10001_100012_clean.dcm",
        "10005_100051_clean.dcm",
    ]

    for p in patients:
        plot_patches_from_hdf5_most(OUTPUT_HDF5, p, save_path_prefix=p.replace(".", "_"))