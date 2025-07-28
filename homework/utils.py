# utils.py

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        """
        Dataset for classification task.
        Args:
            dataset_path: Path to dataset directory
            transform: torchvision transform to apply to images
        """
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    image = Image.open(Path(dataset_path, fname)).convert("RGB")
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((image, label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label


def load_data(
    dataset_path: str,
    num_workers: int = 0,
    batch_size: int = 128,
    shuffle: bool = False,
    transform=None,
) -> DataLoader:
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


def get_transform(mode="basic"):
    if mode == "augment":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.ToTensor()


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return (outputs_idx == labels).float().mean()
