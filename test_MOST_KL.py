import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt

from config import build_config
from dataset import KneeMILDataset
from losses import coral_multitask_predict, coral_predict

from myutils import (
    get_criterion,
    create_transforms,
    get_model,
)


H5_FILE = "MOST_M00_knee_patches_16_100.h5"
NPZ_FILE = "MOST_M00_shapes_LR.npz"

# Metadata datasets stored at the top level of the HDF5 file (not knee groups)
EXCLUDED_KEYS = {"subject_ids", "image_ids", "patch_point_indices"}

PATH_TO_MODEL_DIRECTORY = "original_data/V00/train_KL_only"


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def most_collate_fn(batch):
    """
    Collate function for the MOST dataset.

    The default `mil_collate_fn` stacks each sample's `aux_feature` tensor, but
    in the MOST HDF5 file those tensors have inconsistent shapes (some are
    [1, 1, 0], others [0]) because `aux_feature` is essentially unused for MOST.
    """
    patch_bags = []
    labels = []
    ids = []
    features = []

    for item in batch:
        if len(item) == 4:
            item_patches, item_label, item_id, item_feature = item
        else:
            raise ValueError(
                "Each sample in batch must be a (patches, label, id, feature) tuple."
            )

        if not item_patches:
            print(f"Warning: Empty patch list for ID {item_id}")
            continue

        if isinstance(item_patches, list) and len(item_patches) > 0:
            current_bag = torch.stack(item_patches, dim=0)
        elif torch.is_tensor(item_patches) and item_patches.ndim == 4:
            current_bag = item_patches
        else:
            continue

        patch_bags.append(current_bag)
        labels.append(item_label)
        ids.append(item_id)
        features.append(item_feature)

    if not labels:
        return None, None, None, None

    labels_batch_tensor = torch.stack(labels, dim=0)

    # Do NOT stack features (inconsistent shapes for MOST); return as a list.
    return patch_bags, labels_batch_tensor, ids, features


def run_epoch(loader, model, device, config):

    model.eval()

    # Prepare prediction containers
    if config.multitask_type == "off":
        all_preds = []
        all_labels = []
    else:
        tasks = list(config.OARSI_TASKS.keys())
        all_preds = {task: [] for task in tasks}
        all_labels = {task: [] for task in tasks}

    for list_of_patch_bags, labels_batch, group_names, list_of_features in loader:
        if not list_of_patch_bags:
            continue

        moved_bags = []
        valid_indices = []

        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device, non_blocking=True))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(device, non_blocking=True)

        with torch.no_grad():
            outputs, _, _, _ = model(moved_bags)

        # Predictions
        if config.multitask_type == "off":
            if config.predict_criteria == "Coral":
                predicted, _ = coral_predict(outputs)
            else:  # Max
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
        else:
            # Multi-task
            if config.predict_criteria == "Coral_Multitask":
                predicted, _ = coral_multitask_predict(outputs)
                for task in tasks:
                    pred = predicted[task][0]
                    all_preds[task].extend(pred.cpu().numpy())
                    if task == "kl":
                        all_labels[task].extend(labels_batch.cpu().numpy())
                    else:
                        all_labels[task].extend([-999] * len(pred))
            else:  # Max_Multitask
                for task in tasks:
                    _, pred = torch.max(outputs[task].data, 1)
                    all_preds[task].extend(pred.cpu().numpy())
                    if task == "kl":
                        all_labels[task].extend(labels_batch.cpu().numpy())
                    else:
                        all_labels[task].extend([-999] * len(pred))

    return all_labels, all_preds


def main(config):

    # --------------------------------------------------
    # Load all HDF5 group names (whole dataset is the test set)
    # --------------------------------------------------
    with h5py.File(H5_FILE, "r") as hf:
        all_keys = list(hf.keys())

    test_pids = [k for k in all_keys if k not in EXCLUDED_KEYS]

    print(f"All MOST groups (before KL filtering): {len(test_pids)}")

    # --------------------------------------------------
    # Mean / std
    # --------------------------------------------------
    mean, std =  np.load("original_data/V00/train_KL_only/mean_std_train_patches.npy")#np.load(config.MEAN_STD_FILE_PATH_Optional)
    _, val_transform = create_transforms(mean, std)

    # --------------------------------------------------
    # Dataset (KneeMILDataset filters out KL in {-999, 8, 9})
    # --------------------------------------------------
    test_ds = KneeMILDataset(H5_FILE, test_pids, transform=val_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=most_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(f"Testing samples (after KL filtering): {len(test_ds)}")

    # ======================================================================
    #                           K-FOLD INFERENCE
    # ======================================================================
    config.K_FOLDS = 5

    metrics_per_task = {}   # {"kl": {"acc": [], "f1": [], "kappa": []}}
    agg_cm_raw = None       # accumulated confusion matrix for KL

    for fold in range(config.K_FOLDS):

        print(f"\n========== Inference Fold {fold+1}/{config.K_FOLDS} ==========")

        # ----------------- Model ----------------- #
        model = get_model(config)
        model.to(config.DEVICE)

        ckpt_name = f"best_model_avg_kappa_fold{fold+1}.pth"
        ckpt_path = os.path.join(PATH_TO_MODEL_DIRECTORY, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Missing checkpoint: {ckpt_name}, skipping this fold.")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

        # ----------------- Run inference ----------------- #
        test_labels, test_preds = run_epoch(
            test_loader, model, config.DEVICE, config,
        )

        # ----------------- KL metrics ----------------- #
        if config.multitask_type == "off":
            labels = [int(l) for l in test_labels]
            preds = [int(p) for p in test_preds]
        else:
            labels = [int(l) for l in test_labels["kl"]]
            preds = [int(p) for p in test_preds["kl"]]

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        kappa = cohen_kappa_score(labels, preds, weights="quadratic")

        print(f"[Fold {fold+1}] KL - Acc={acc:.4f} F1(macro)={f1:.4f} Kappa={kappa:.4f}")

        report = classification_report(labels, preds, zero_division=0)

        # Save report
        report_path = os.path.join(config.CHECKPOINT_DIR, f"classification_kl_fold{fold+1}.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
            f.write("\n")
            f.write(f"Acc   = {acc:.4f}\n")
            f.write(f"F1    = {f1:.4f}\n")
            f.write(f"Kappa = {kappa:.4f}\n")

        # Accumulate raw confusion matrix (sum across folds)
        fold_cm_raw = confusion_matrix(labels, preds)
        if agg_cm_raw is None:
            agg_cm_raw = fold_cm_raw.copy()
        else:
            agg_cm_raw += fold_cm_raw

        # Confusion matrix
        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })
        disp = ConfusionMatrixDisplay.from_predictions(
            labels, preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f'
        )
        ax = disp.ax_
        for text in ax.texts:
            text.set_fontsize(16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        plt.tight_layout()
        cm_path = os.path.join(config.CHECKPOINT_DIR, f"cm_kl_fold{fold+1}.png")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        # Store for final stats
        if "kl" not in metrics_per_task:
            metrics_per_task["kl"] = {"acc": [], "f1": [], "kappa": []}
        metrics_per_task["kl"]["acc"].append(acc)
        metrics_per_task["kl"]["f1"].append(f1)
        metrics_per_task["kl"]["kappa"].append(kappa)

        # Save predictions for this fold
        np.savez(
            os.path.join(config.CHECKPOINT_DIR, f"test_pred_fold{fold+1}.npz"),
            id=test_ds.sample_group_names,
            pred=preds,
            true=labels,
        )

        plt.close('all')

    # ----------------- AGGREGATED CONFUSION MATRIX ----------------- #
    if agg_cm_raw is not None:
        cm_norm = agg_cm_raw.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

        num_classes = cm_norm.shape[0]
        class_names = [str(i) for i in range(num_classes)]

        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })
        disp = ConfusionMatrixDisplay(
            cm_norm, display_labels=class_names
        )
        disp.plot(cmap=plt.cm.Greens, values_format='.2f')
        ax = disp.ax_
        for text in ax.texts:
            text.set_fontsize(16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        plt.tight_layout()
        cm_path = os.path.join(config.CHECKPOINT_DIR, "cm_kl_aggregated.png")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path)
        plt.close()
        print("[INFO] Aggregated confusion matrix saved as cm_kl_aggregated.png")

    # ----------------- CROSS-FOLD STATISTICAL SUMMARY ----------------- #
    print("\n===== Cross-fold Statistical Summary =====")

    summary_path = os.path.join(config.CHECKPOINT_DIR, "metrics_summary_kappa.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        for task, vals in metrics_per_task.items():
            acc_arr = np.array(vals["acc"])
            f1_arr = np.array(vals["f1"])
            kappa_arr = np.array(vals["kappa"])

            acc_mean, acc_std = acc_arr.mean(), acc_arr.std()
            f1_mean, f1_std = f1_arr.mean(), f1_arr.std()
            kappa_mean, kappa_std = kappa_arr.mean(), kappa_arr.std()

            print(f"\nTask: {task}")
            print(f"  Acc   = {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  F1    = {f1_mean:.4f} ± {f1_std:.4f}")
            print(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}")

            f.write(f"Task: {task}\n")
            f.write(f"  Acc   = {acc_mean:.4f} ± {acc_std:.4f}\n")
            f.write(f"  F1    = {f1_mean:.4f} ± {f1_std:.4f}\n")
            f.write(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}\n\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\nTesting finished.")


if __name__ == "__main__":

    config_dict = build_config()

    cfg = Config(config_dict)

    cfg.WANDB = False

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    cfg.model_type = "MIL"
    cfg.lossfcn_type = "CrossEntropy"
    cfg.predict_criteria = "Max"
    cfg.classweight_type = "inv"
    cfg.multitask_type = "off"
    cfg.note = "demo"
    # build_config only sets the full 7-task dict when multitask_type=="all"
    # is parsed from argv, so set it explicitly here since we override above. #"jsnm": 4, "jsnl": 4, "osfm": 4, "ostm": 4, "ostl": 4, "osfl": 4
    cfg.OARSI_TASKS = {
        "kl": 5, 
    }

    main(cfg)