# train_detection.py

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.utils.tensorboard as tb

from models import Detector, save_model
from metrics import ConfusionMatrix
from datasets.road_dataset import load_data


def compute_metrics(seg_logits, seg_labels, pred_depth, gt_depth, cm: ConfusionMatrix):
    """
    Updates ConfusionMatrix and computes depth MAE and lane-only MAE.
    """
    preds = torch.argmax(seg_logits, dim=1)
    cm.add(preds, seg_labels)

    abs_err = torch.abs(pred_depth - gt_depth)
    mae = abs_err.mean().item()

    # Only consider left/right track pixels (class 1 or 2)
    lane_mask = seg_labels != 0
    if lane_mask.any():
        lane_mae = abs_err[lane_mask].mean().item()
    else:
        lane_mae = 0.0

    return mae, lane_mae


def train(
    exp_dir="logs",
    num_epoch=30,
    lr=1e-3,
    batch_size=64,
    seed=2024,
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging
    log_dir = Path(exp_dir) / f"detection_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Model + optimizer
    model = Detector().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_seg = nn.CrossEntropyLoss()
    loss_depth = nn.L1Loss()

    # Data
    train_loader = load_data("../drive_data/train", batch_size=batch_size, shuffle=True)
    val_loader = load_data("../drive_data/val", batch_size=batch_size, shuffle=False)

    for epoch in range(num_epoch):
        model.train()
        cm_train = ConfusionMatrix(num_classes=3)
        train_mae_list, train_lane_mae_list = [], []

        for batch in train_loader:
            img = torch.tensor(batch["image"]).to(device)
            track = torch.tensor(batch["track"]).to(device).long()
            depth = torch.tensor(batch["depth"]).to(device).float()

            optimizer.zero_grad()
            seg_logits, pred_depth = model(img)

            loss = loss_seg(seg_logits, track) + loss_depth(pred_depth, depth)
            loss.backward()
            optimizer.step()

            mae, lane_mae = compute_metrics(
                seg_logits, track, pred_depth, depth, cm_train
            )
            train_mae_list.append(mae)
            train_lane_mae_list.append(lane_mae)

        train_miou = cm_train.iou().mean().item()
        train_mae = np.mean(train_mae_list)
        train_lane_mae = np.mean(train_lane_mae_list)

        # Validation
        model.eval()
        cm_val = ConfusionMatrix(num_classes=3)
        val_mae_list, val_lane_mae_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                img = torch.tensor(batch["image"]).to(device)
                track = torch.tensor(batch["track"]).to(device).long()
                depth = torch.tensor(batch["depth"]).to(device).float()

                seg_logits, pred_depth = model(img)
                mae, lane_mae = compute_metrics(
                    seg_logits, track, pred_depth, depth, cm_val
                )
                val_mae_list.append(mae)
                val_lane_mae_list.append(lane_mae)

        val_miou = cm_val.iou().mean().item()
        val_mae = np.mean(val_mae_list)
        val_lane_mae = np.mean(val_lane_mae_list)

        # Logging
        logger.add_scalar("train/miou", train_miou, epoch)
        logger.add_scalar("train/mae", train_mae, epoch)
        logger.add_scalar("train/lane_mae", train_lane_mae, epoch)
        logger.add_scalar("val/miou", val_miou, epoch)
        logger.add_scalar("val/mae", val_mae, epoch)
        logger.add_scalar("val/lane_mae", val_lane_mae, epoch)

        print(
            f"[Epoch {epoch+1:02d}] "
            f"train_mIoU={train_miou:.4f} | val_mIoU={val_miou:.4f} | "
            f"val_MAE={val_mae:.4f} | val_laneMAE={val_lane_mae:.4f}"
        )

    save_model(model)
    torch.save(model.state_dict(), log_dir / "detector.th")
    print(f"Model saved to {log_dir / 'detector.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    train(**vars(args))
