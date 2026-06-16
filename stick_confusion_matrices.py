from PIL import Image
import matplotlib.pyplot as plt
import math
import os

folder = "original_data/V00/model_train_with_testset_OAI_5fold_kl+OARIS"

image_paths = [
    f"{folder}/cm_jsnl.png",
    f"{folder}/cm_jsnm.png",
    f"{folder}/cm_osfl.png",
    f"{folder}/cm_ostm.png",
    f"{folder}/cm_osfm.png",
    f"{folder}/cm_ostl.png",
    f"{folder}/cm_kl.png",
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