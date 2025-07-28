# train_classification.py

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.utils.tensorboard as tb

from models import load_model, save_model, ClassificationLoss
from datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ClassificationLoss()

    train_data = load_data(
        "../classification_data/train",
        shuffle=True,
        batch_size=batch_size,
        transform_pipeline="aug",
    )

    val_data = load_data("../classification_data/val", shuffle=False)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(acc)

            global_step += 1

        model.eval()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == label).float().mean().item()
                metrics["val_acc"].append(acc)

        train_acc = torch.tensor(metrics["train_acc"]).mean()
        val_acc = torch.tensor(metrics["val_acc"]).mean()

        logger.add_scalar("train/accuracy", train_acc, epoch)
        logger.add_scalar("val/accuracy", val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
