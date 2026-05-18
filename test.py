import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config import build_config
from dataset import KneeMILDataset, mil_collate_fn
from losses import coral_predict, coral_multitask_predict
from myutils import (
    compute_metrics,
    get_criterion,
    labels_to_levels,
    create_transforms,
    get_model,
    get_model_org,
    build_CAM_attention_tool,
)


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def run_epoch(loader, model, model_org, criterion, device, config, desc=""):
    model.eval()

    total_loss = 0.0
    num_processed_samples = 0

    if config.multitask_type == "off":
        all_preds, all_labels = [], []
    else:
        all_preds = {task: [] for task in config.OARSI_TASKS.keys()}
        all_labels = {task: [] for task in config.OARSI_TASKS.keys()}

    attention_tool = build_CAM_attention_tool(config.feedback_cam, model_org) if model_org else None

    if model_org:
        model_org.eval()

    for list_of_patch_bags, labels_batch, group_name, list_of_features in loader:
        if not list_of_patch_bags:
            continue

        moved_bags, moved_features, valid_indices = [], [], []

        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device, non_blocking=True))
                moved_features.append(list_of_features[i][0].to(device, non_blocking=True))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(device, non_blocking=True)

        with torch.no_grad():
            if config.feedback_type == "off":
                outputs, _, _, _ = model(moved_bags)
            else:
                outputs, _, _, _ = model(moved_bags, model_org, attention_tool)

            # Targets
            if config.multitask_type == "off":
                loss = criterion(outputs, labels_batch)

            else:
                if config.multitask_type == "all":
                    targets = {
                        "kl": labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                        "osfm": torch.tensor([f[2] for f in moved_features], device=device),
                        "ostm": torch.tensor([f[3] for f in moved_features], device=device),
                        "ostl": torch.tensor([f[4] for f in moved_features], device=device),
                        "osfl": torch.tensor([f[5] for f in moved_features], device=device),
                    }

                    for k, v in targets.items():
                        targets[k] = torch.where(v == -999, torch.tensor(0, device=device), v)

                elif config.multitask_type == "kl_jsn":
                    targets = {
                        "kl": labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                    }

                if config.lossfcn_type == "CoralLoss_MultiTask":
                    targets_levels = {}
                    for k, v in targets.items():
                        num_classes = config.OARSI_TASKS[k]
                        targets_levels[k] = labels_to_levels(v, num_classes)

                    loss, _ = criterion(outputs, targets_levels)
                else:
                    loss, _ = criterion(outputs, targets)

        total_loss += loss.item() * labels_batch.size(0)
        num_processed_samples += labels_batch.size(0)

        # Predictions
        if config.multitask_type == "off":
            if config.predict_criteria == "Coral":
                predicted, _ = coral_predict(outputs)
            else:
                _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

        else:
            if config.predict_criteria == "Coral_Multitask":
                predicted, _ = coral_multitask_predict(outputs)

                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task][0].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())

            else:
                predicted = {}

                for task, out in outputs.items():
                    _, pred = torch.max(out.data, 1)
                    predicted[task] = pred

                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0

    return avg_loss, all_labels, all_preds



def main(config):
    # Load split
    with open("splits.json", "r") as f:
        splits = json.load(f)

    test_pids = splits["test"]

    if "9491446_R" in test_pids:
        test_pids.remove("9491446_R")

    print(f"Testing samples: {len(test_pids)}")

    # Mean / std
    mean, std = np.load(config.MEAN_STD_FILE_PATH_Optional)
    _, val_transform = create_transforms(mean, std)

    # Dataset
    test_ds = KneeMILDataset(
        config.H5_FILE,
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

    # Model
    model = get_model(config)
    model_org = get_model_org(config)

    checkpoint_path = os.path.join(
        config.PRETRAINED_MODEL_PATH,
        f"best_model_{config.inference_target}_kappa.pth"
    )

    print(f"Loading checkpoint: {checkpoint_path}")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=config.DEVICE)
    )

    model.to(config.DEVICE)

    if model_org:
        model_org.load_state_dict(
            torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
        )
        model_org.to(config.DEVICE)

    # Criterion
    criterion = get_criterion(
        config.lossfcn_type,
        None,
        config.OARSI_TASKS
    )

    # Test
    test_loss, test_labels, test_preds = run_epoch(
        test_loader,
        model,
        model_org,
        criterion,
        config.DEVICE,
        config,
        desc="[Test]"
    )

    print(f"\nTest Loss: {test_loss:.4f}")

    test_metrics = compute_metrics(
        config.multitask_type,
        test_labels,
        test_preds
    )

    # Results
    if config.multitask_type == "off":
        print(test_metrics)

        print(classification_report(
            test_labels,
            test_preds,
            zero_division=0
        ))

        plt.figure(figsize=(8, 8))

        ConfusionMatrixDisplay.from_predictions(
            test_labels,
            test_preds,
            normalize="true",
            cmap=plt.cm.Blues,
            values_format=".2f"
        )

        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHECKPOINT_DIR, "cm_test.png"))
        plt.close()

    else:
        for task in test_labels.keys():
            print(f"\n===== {task.upper()} =====")
            print(test_metrics[task])

            print(classification_report(
                test_labels[task],
                test_preds[task],
                zero_division=0
            ))

            plt.figure(figsize=(8, 8))

            ConfusionMatrixDisplay.from_predictions(
                test_labels[task],
                test_preds[task],
                normalize="true",
                cmap=plt.cm.Blues,
                values_format=".2f"
            )

            plt.title(f"{task.upper()} Confusion Matrix")
            plt.tight_layout()

            plt.savefig(
                os.path.join(config.CHECKPOINT_DIR, f"cm_{task}.png")
            )
            plt.close()

    print("\nTesting finished.")


if __name__ == "__main__":
    config_dict = build_config()
    cfg = Config(config_dict)
    cfg.WANDB = False
    main(cfg)
# ```

# Your `splits.json` should look like:

# ```json
# {
#     "train": ["id1_L", "id2_R"],
#     "val": ["id3_L"],
#     "test": ["id4_R", "id5_L"]
# }
# ```

# Then simply run:

# ```bash
# python test.py
# ```
