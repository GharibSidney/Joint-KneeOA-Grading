import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config import build_config
from dataset import KneeMILDataset, mil_collate_fn
from losses import coral_predict
from myutils import (
    compute_metrics,
    get_criterion,
    labels_to_levels,
    create_transforms,
    get_model,
)


H5_FILE = "MOST_M00_knee_patches_16_100.h5"
NPZ_FILE = "MOST_M00_shapes_LR.npz"



class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def run_epoch(loader, model, criterion, device, config):

    model.eval()

    total_loss = 0.0
    num_processed_samples = 0

    all_preds = []
    all_labels = []

    for list_of_patch_bags, labels_batch, group_name, list_of_features in loader:
        if not list_of_patch_bags:
            continue

        moved_bags = []
        valid_indices = []

        for i, bag in enumerate(list_of_patch_bags):

            if bag.nelement() > 0:

                moved_bags.append(
                    bag.to(device, non_blocking=True)
                )

                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(
            device,
            non_blocking=True
        )

        with torch.no_grad():

            outputs, _, _, _ = model(moved_bags)

            if config.lossfcn_type == "CoralLoss":

                labels_levels = labels_to_levels(
                    labels_batch,
                    config.NUM_CLASSES
                )

                loss = criterion(outputs, labels_levels)

            else:

                loss = criterion(outputs, labels_batch)

        total_loss += loss.item() * labels_batch.size(0)

        num_processed_samples += labels_batch.size(0)

        # Predictions
        if config.predict_criteria == "Coral":

            predicted, _ = coral_predict(outputs)

        else:

            _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    avg_loss = (
        total_loss / num_processed_samples
        if num_processed_samples > 0 else 0
    )

    return avg_loss, all_labels, all_preds


def main(config):

    # --------------------------------------------------
    # Load all HDF5 group names
    # --------------------------------------------------
    with h5py.File(H5_FILE, "r") as hf:

        all_keys = list(hf.keys())

    # Remove metadata datasets
    excluded = { "subject_ids", "image_ids", "patch_point_indices"}

    test_pids = [
        k for k in all_keys
        if k not in excluded
    ]

    print(f"Testing samples: {len(test_pids)}")

    # --------------------------------------------------
    # Mean / std
    # --------------------------------------------------
    mean, std = np.load(
        config.MEAN_STD_FILE_PATH_Optional
    )

    _, val_transform = create_transforms(
        mean,
        std
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    test_ds = KneeMILDataset(
        H5_FILE,
        test_pids,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    # After test_loader is created, before run_epoch
    for _, labels_batch, _, _ in test_loader:
        print(f"Unique labels : {labels_batch.unique().tolist()}")
        print(f"Min / Max     : {labels_batch.min().item()} / {labels_batch.max().item()}")
        break  # just check the first batch

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
        torch.load(
            checkpoint_path,
            map_location=config.DEVICE
        )
    )

    model.to(config.DEVICE)
    # Find the last linear layer and check its output size
    for name, module in model.named_modules():
        if hasattr(module, 'out_features'):
            print(f"{name}: out_features = {module.out_features}")
    # --------------------------------------------------
    # Criterion
    # --------------------------------------------------
    criterion = get_criterion(
        config.lossfcn_type,
        None,
        None
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    test_loss, test_labels, test_preds = run_epoch(
        test_loader,
        model,
        criterion,
        config.DEVICE,
        config,
    )
    print("\nPredictions:")
    for gt, pred in zip(test_labels, test_preds):
        print(f"GT: {gt} | Pred: {pred}")

    print(f"\nTest Loss: {test_loss:.4f}")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    test_metrics = compute_metrics(
        "off",
        test_labels,
        test_preds
    )

    print("\nMetrics:")
    print(test_metrics)

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    plt.figure(figsize=(8, 8))

    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        test_preds,
        normalize="true",
        cmap=plt.cm.Blues,
        values_format=".2f"
    )

    plt.title("MOST Test Confusion Matrix")

    plt.tight_layout()

    cm_path = os.path.join(
        config.CHECKPOINT_DIR,
        "cm_MOST_test.png"
    )

    plt.savefig(cm_path)

    plt.close()

    print(f"\nConfusion matrix saved to: {cm_path}")

    print("\nTesting finished.")


if __name__ == "__main__":

    config_dict = build_config()

    cfg = Config(config_dict)

    cfg.WANDB = False

    main(cfg)