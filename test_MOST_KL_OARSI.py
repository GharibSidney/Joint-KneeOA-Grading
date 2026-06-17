import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config import build_config
from dataset import KneeMILDataset
from losses import coral_multitask_predict

from myutils import (
    compute_metrics,
    get_criterion,
    labels_to_levels,
    create_transforms,
    get_model,
)


H5_FILE = "MOST_M00_knee_patches_16_100.h5"
NPZ_FILE = "MOST_M00_shapes_LR.npz"

# Metadata datasets stored at the top level of the HDF5 file (not knee groups)
EXCLUDED_KEYS = {"subject_ids", "image_ids", "patch_point_indices"}

# Mapping from the OARSI datasets stored inside each MOST knee group
# (see shape_patch_MOST.py) to the model's multitask head names.
OARSI_FIELD_TO_TASK = {
    "joint_space_medial":        "jsnm",
    "joint_space_lateral":       "jsnl",
    "osteophyte_femur_medial":   "osfm",
    "osteophyte_tibia_medial":   "ostm",
    "osteophyte_tibia_lateral":  "ostl",
    "osteophyte_femur_lateral":  "osfl",
}

# Valid label ranges. Anything outside of these (including -999) is dropped
# before computing metrics for that particular task.
VALID_KL_LABELS = {0, 1, 2, 3, 4}
VALID_OARSI_LABELS = {0, 1, 2, 3}


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
    The OARSI targets are read separately (see `load_oarsi_targets`), so here we
    simply skip stacking the features and return them as a list instead.
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



def load_oarsi_targets(h5_file, group_names):
    """
    Read the 6 OARSI scalar datasets for each knee group directly from the
    MOST HDF5 file (they are NOT stored in `aux_feature` for MOST).

    Returns a dict: group_name -> {task: int_value}
    """
    oarsi_by_group = {}
    with h5py.File(h5_file, "r") as hf:
        for name in group_names:
            grp = hf[name]
            entry = {}
            for field, task in OARSI_FIELD_TO_TASK.items():
                if field in grp:
                    entry[task] = int(np.array(grp[field]).flatten()[0])
                else:
                    entry[task] = -999
            oarsi_by_group[name] = entry
    return oarsi_by_group


def run_epoch(loader, model, criterion, device, config, oarsi_by_group):

    model.eval()

    tasks = list(config.OARSI_TASKS.keys())  # e.g. ["kl", "jsnm", "jsnl", "osfm", "ostm", "ostl", "osfl"]

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
        valid_group_names = [group_names[i] for i in valid_indices]

        # Build the per-task targets for this batch.
        targets = {"kl": labels_batch}
        for task in tasks:
            if task == "kl":
                continue
            task_vals = [
                oarsi_by_group[name][task] for name in valid_group_names
            ]
            targets[task] = torch.tensor(task_vals, device=device)

        with torch.no_grad():
            outputs, _, _, _ = model(moved_bags)

        # Predictions per task
        if config.predict_criteria == "Coral_Multitask":
            predicted, _ = coral_multitask_predict(outputs)
            for task in tasks:
                pred = predicted[task][0]
                all_preds[task].extend(pred.cpu().numpy())
                all_labels[task].extend(targets[task].cpu().numpy())
        else:  # Max_Multitask
            for task in tasks:
                _, pred = torch.max(outputs[task].data, 1)
                all_preds[task].extend(pred.cpu().numpy())
                all_labels[task].extend(targets[task].cpu().numpy())

    return all_labels, all_preds


def filter_valid_pairs(task, labels, preds):
    """
    Keep only (label, pred) pairs whose ground-truth label is valid for the
    given task. KL labels must be in {0..4}; OARSI labels must be in {0..3}.
    Missing (-999) or out-of-range values are dropped.
    """
    valid_set = VALID_KL_LABELS if task == "kl" else VALID_OARSI_LABELS
    f_labels, f_preds = [], []
    for gt, pred in zip(labels, preds):
        if int(gt) in valid_set:
            f_labels.append(int(gt))
            f_preds.append(int(pred))
    return f_labels, f_preds


def main(config):

    if config.multitask_type != "all":
        raise ValueError(
            "test_MOST_KL&OARSI.py must be run with --multitask_type all "
            f"(got '{config.multitask_type}')."
        )

    # --------------------------------------------------
    # Load all HDF5 group names (whole dataset is the test set)
    # --------------------------------------------------
    with h5py.File(H5_FILE, "r") as hf:
        all_keys = list(hf.keys())

    test_pids = [k for k in all_keys if k not in EXCLUDED_KEYS]

    print(f"Testing samples (before KL filtering): {len(test_pids)}")

    # --------------------------------------------------
    # Mean / std
    # --------------------------------------------------
    mean, std = np.load(config.MEAN_STD_FILE_PATH_Optional)
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

    # Pre-load OARSI targets for the groups that survived KL filtering
    oarsi_by_group = load_oarsi_targets(H5_FILE, test_ds.sample_group_names)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = get_model(config)

    checkpoint_path = os.path.join(
        config.PRETRAINED_MODEL_PATH,
        "best_model_kl_kappa.pth"
    )
    print(f"Loading checkpoint: {checkpoint_path}")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)

    # --------------------------------------------------
    # Criterion (kept for parity with other scripts; not used for scoring)
    # --------------------------------------------------
    criterion = get_criterion(config.lossfcn_type, None, config.OARSI_TASKS)

    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    test_labels, test_preds = run_epoch(
        test_loader, model, criterion, config.DEVICE, config, oarsi_by_group
    )

    # --------------------------------------------------
    # Metrics / reports / confusion matrices (per task, valid values only)
    # --------------------------------------------------
    results = ["=== MOST KL & OARSI Inference ==="]

    for task in config.OARSI_TASKS.keys():
        labels, preds = filter_valid_pairs(task, test_labels[task], test_preds[task])

        if len(labels) == 0:
            line = f"[{task}] No valid samples - skipped."
            print(line)
            results.append(line)
            continue

        metrics = compute_metrics("off", labels, preds)["kl"]
        acc = metrics["acc"]
        f1 = metrics["f1"]
        kappa = metrics["kappa"]

        header = (
            f"[Test] {task.upper()} (n={len(labels)}) - "
            f"Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}"
        )
        print("\n" + header)
        results.append(header)

        num_classes = config.OARSI_TASKS[task]
        report = classification_report(labels, preds, zero_division=0)
        print(report)
        results.append(f"\n[{task}]\n" + report)

        # Confusion matrix
        plt.figure(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            labels,
            preds,
            normalize="true",
            cmap=plt.cm.Blues,
            values_format=".2f",
        )
        plt.title(f"MOST {task.upper()} - Normalized Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(config.CHECKPOINT_DIR, f"cm_MOST_{task}.png")

        # Create parent directory of the output file
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)

        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

        # Save per-task predictions
        np.savez(
            os.path.join(config.CHECKPOINT_DIR, f"test_pred_MOST_{task}.npz"),
            pred=np.array(preds),
            true=np.array(labels),
        )

    # --------------------------------------------------
    # Save text summary
    # --------------------------------------------------
    save_path = os.path.join(config.CHECKPOINT_DIR, "MOST_inference_result.txt")
    with open(save_path, "w") as f:
        for line in results:
            f.write(line + "\n")
    print(f"\nResults written to: {save_path}")

    print("\nTesting finished.")


if __name__ == "__main__":

    config_dict = build_config()

    cfg = Config(config_dict)

    cfg.WANDB = False

    # ------------------------------------------------------------------
    # Default run configuration baked in so that a bare
    #   python test_MOST_KL_OARSI.py --current_ckpt [MODEL_FOLDER]
    # behaves as if the following flags were passed:
    #   --model_type MIL_MultiTask_imedslab
    #   --lossfcn_type CrossEntropy_MultiTask
    #   --predict_criteria Max_Multitask
    #   --classweight_type all_metrics_inv
    #   --multitask_type all
    #   --note demo
    # ------------------------------------------------------------------
    cfg.model_type = "MIL_MultiTask_imedslab"
    cfg.lossfcn_type = "CrossEntropy_MultiTask"
    cfg.predict_criteria = "Max_Multitask"
    cfg.classweight_type = "all_metrics_inv"
    cfg.multitask_type = "all"
    cfg.note = "demo"
    # build_config only sets the full 7-task dict when multitask_type=="all"
    # is parsed from argv, so set it explicitly here since we override above.
    cfg.OARSI_TASKS = {
        "kl": 5, "jsnm": 4, "jsnl": 4, "osfm": 4, "ostm": 4, "ostl": 4, "osfl": 4
    }

    main(cfg)


