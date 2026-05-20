import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_patches_from_hdf5(hdf5_path, patient_id, side, n_cols=8, save_path=None):
    with h5py.File(hdf5_path, 'r') as hf:
        group_name = f"{patient_id}_{side}"
        grp = hf[group_name]
        patches = grp['patches'][:]
        kl = grp['kl_grade'][0]
        aux = grp['aux_feature'][0]
        indices = grp['patch_source_point_indices'][:]

    patches = patches[..., 0]
    n_patches = len(patches)
    n_rows = math.ceil(n_patches / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for idx, patch in enumerate(patches):
        axes[idx].imshow(patch, cmap='gray')
        axes[idx].set_title(f"pt {indices[idx]}", fontsize=8)
        axes[idx].axis('off')

    for idx in range(n_patches, len(axes)):
        axes[idx].axis('off')

    knee_label = "Left (flipped)" if side == "L" else "Right"
    plt.suptitle(f"Patient {patient_id} — {knee_label} knee | KL={kl} | Aux={aux}")
    plt.tight_layout()

    out = save_path or f"{patient_id}_{side}_patches.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out}")

if __name__ == "__main__":
    OUTPUT_HDF5_PATIENT_GROUPED_FILE = "original_data/V00/V00_knee_patches_patient_grouped_16_100_all_feature.h5"
    patient_id = "9008884"
    plot_patches_from_hdf5(OUTPUT_HDF5_PATIENT_GROUPED_FILE, patient_id, "L", save_path="patches_L.png")
    plot_patches_from_hdf5(OUTPUT_HDF5_PATIENT_GROUPED_FILE, patient_id, "R", save_path="patches_R.png")