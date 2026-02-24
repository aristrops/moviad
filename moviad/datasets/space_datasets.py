import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from pathlib import Path


def show_mars(path):
    img = np.load(path)

    if img.shape[-1] == 6:
        img = np.transpose(img, (2, 0, 1))

    rgb = np.stack([
        img[2],
        img[0],
        img[1]
    ], axis=-1)

    # normalize for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    plt.imshow(rgb)
    plt.title("False Color Composite")
    plt.axis("off")
    plt.show()


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    mean = 0
    std = 0
    total = 0

    for imgs, _ in loader:
        b, c, h, w = imgs.shape
        pixels = b * h * w

        mean += imgs.mean(dim=[0, 2, 3]) * pixels
        std += imgs.std(dim=[0, 2, 3]) * pixels
        total += pixels

    mean /= total
    std /= total

    return mean, std


class MarsDataset(Dataset):
    def __init__(self, root_dir, split="all", transform=None, test_positive_ratio=None):
        self.samples = []
        self.transform = None

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
        else:
            self.transform = transform

        root = Path(root_dir)

        split_map = {
            "train": [
                (root / "train_typical", 0),
            ],
            "test": [
                (root / "test_typical", 0),
                (root / "test_novel", 1),
            ],
            "all": [
                (root / "train_typical", 0),
                (root / "test_typical", 0),
                (root / "test_novel", 1),
            ],
        }

        if split not in split_map:
            raise ValueError(f"Invalid split: {split}")

        for folder, label in split_map[split]:
            if not folder.exists():
                continue

            for f in folder.rglob("*.npy"):
                self.samples.append((str(f), label))

        if split == "test" and test_positive_ratio is not None:
            negatives = [s for s in self.samples if s[1] == 0]
            positives = [s for s in self.samples if s[1] == 1]

            target_pos = int(len(negatives) * test_positive_ratio / (1 - test_positive_ratio))

            positives = random.sample(positives, min(target_pos, len(positives)))

            self.samples = negatives + positives
            random.shuffle(self.samples)

            print(f"Test set balanced to {test_positive_ratio * 100:.1f}% positives")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.load(path)

        # ensure channel-first
        if img.shape[-1] == 6:
            img = np.transpose(img, (2, 0, 1))

        img = np.stack([img[2], img[0], img[1]], axis=0)  # RGB ordered channels

        img = torch.from_numpy(img).float()

        if self.transform:
            img = self.transform(img)

        return img, label


class LunarDataset(Dataset):
    def __init__(self, root_dir=None, split="all", transform=None, test_positive_ratio=None):
        self.samples = []
        self.transform = None

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        root = Path(root_dir)

        split_map = {
            "train": [
                (root / "none_train1", 0),
                (root / "none_train2", 0),
                (root / "none_train3", 0),
                (root / "none_train4", 0),
            ],
            "test": [
                (root / "test/test/none", 0),
                (root / "test/test/oldcrater", 1),
                (root / "test/test/ejecta", 1),
            ],
            "all": [
                (root / "none_train1", 0),
                (root / "none_train2", 0),
                (root / "none_train3", 0),
                (root / "none_train4", 0),
                (root / "test/none", 0),
                (root / "test/oldcrater", 1),
                (root / "test/ejecta", 1),
            ],
        }

        for folder, label in split_map[split]:
            if not folder.exists() or "MACOSX" in str(folder):
                continue

            for f in folder.rglob("*.jpg"):
                if "__MACOSX" in f.parts:
                    continue

                # if f.name.startswith("._") or f.name.startswith("."):
                #     continue
                #
                #     # Only allow real images
                # if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                #     continue

                self.samples.append((str(f), label))

        if split == "test" and test_positive_ratio is not None:
            negatives = [s for s in self.samples if s[1] == 0]
            positives = [s for s in self.samples if s[1] == 1]

            target_pos = int(len(negatives) * test_positive_ratio / (1 - test_positive_ratio))

            positives = random.sample(positives, min(target_pos, len(positives)))

            self.samples = negatives + positives
            random.shuffle(self.samples)

            print(f"Test set balanced to {test_positive_ratio * 100:.1f}% positives")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":

    mars_mean, mars_std = compute_mean_std(
        MarsDataset(root_dir="/home/fgenilotti/Downloads/vad-space/mars/3732485", split="train", transform=None))

    from torchvision import transforms

    mars_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
    ])

    lunar_dataset = LunarDataset(root_dir="/home/fgenilotti/Downloads/vad-space/lunar/7041842", split="test",
                                 transform=None, test_positive_ratio=0.05)
    mars_dataset = MarsDataset(root_dir="/home/fgenilotti/Downloads/vad-space/mars/3732485", split="train",
                               transform=mars_transform)

    lunar_loader = DataLoader(lunar_dataset, batch_size=32, shuffle=False)
    mars_loader = DataLoader(mars_dataset, batch_size=32, shuffle=False)

    for x, label in mars_loader:
        continue

    for x, label in lunar_loader:
        continue