from PIL import Image
import matplotlib.pyplot as plt
import math
import os

folder= "original_data/V00/model_checkpoints_20260708_125959_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16"
tested_MOST = False
if tested_MOST:
        image_paths = [
        f"{folder}/cm_MOST_jsnl.png",
        f"{folder}/cm_MOST_jsnm.png",
        f"{folder}/cm_MOST_osfl.png",
        f"{folder}/cm_MOST_ostm.png",
        f"{folder}/cm_MOST_osfm.png",
        f"{folder}/cm_MOST_ostl.png",
        f"{folder}/cm_MOST_kl.png",
    ]
else:
    image_paths = [
        f"{folder}/cm_jsnl_aggregated.png",
        f"{folder}/cm_jsnm_aggregated.png",
        f"{folder}/cm_osfl_aggregated.png",
        f"{folder}/cm_ostm_aggregated.png",
        f"{folder}/cm_osfm_aggregated.png",
        f"{folder}/cm_ostl_aggregated.png",
        f"{folder}/cm_kl_aggregated.png",
    ]

cols = 2  # 2 graphs per row
rows = math.ceil(len(image_paths) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
axes = axes.flatten()

for ax, path in zip(axes, image_paths):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(os.path.basename(path).replace(".png", ""))
    ax.axis("off")

# Hide unused subplots
for ax in axes[len(image_paths):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()