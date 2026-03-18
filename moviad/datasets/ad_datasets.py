import random
import numpy as np

from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AnoVoxDataset(Dataset):
    def __init__(
            self,
            root_dir,
            mode="all",
            normal_split_ratio=0.8,
            transform=None,
            sem_transform=None
    ):
        assert mode in ["train", "test", "all"]

        self.ANOMALY_COLOR = np.array([245, 0, 0])

        self.samples = []
        self.mode = mode
        self.normal_split_ratio = normal_split_ratio
        self.transform = transform
        self.sem_transform = sem_transform

        self.root_dir = Path(root_dir)
        self._build_index()
        self._apply_vad_split()

    def _contains_anomaly(self, sem_path):
        sem = np.array(Image.open(sem_path))
        anomaly_pixels = np.all(sem == self.ANOMALY_COLOR, axis=-1)
        return int(anomaly_pixels.any())

    def _build_index(self):
        scenario_dirs = sorted(self.root_dir.glob("Scenario_*"))

        for scen_dir in scenario_dirs:
            rgb_dir = scen_dir / "RGB_IMG"
            sem_dir = scen_dir / "SEMANTIC_IMG"

            if not rgb_dir.exists() or not sem_dir.exists():
                continue

            rgb_files = sorted(rgb_dir.glob("*"))

            for rgb_path in rgb_files:
                img_id = rgb_path.name.split('_')[-1]

                sem_name = f"SEMANTIC_IMG_{img_id}"
                sem_path = sem_dir / sem_name

                if sem_path.exists():
                    label = self._contains_anomaly(sem_path)
                    self.samples.append((rgb_path, sem_path, label))

    def _apply_vad_split(self):
        if self.mode == "all":
            return

        normals = [s for s in self.samples if s[-1] == 0]
        anomalies = [s for s in self.samples if s[-1] == 1]

        np.random.shuffle(normals)
        np.random.shuffle(anomalies)

        split_idx = int(len(normals) * self.normal_split_ratio)

        train_normals = normals[:split_idx]
        test_normals = normals[split_idx:]

        if self.mode == "train":
            self.samples = train_normals

        elif self.mode == "test":
            n_test_normals = int(len(normals) * (1 - self.normal_split_ratio))
            n_test_anomalies = int(n_test_normals * 0.2 / (1 - 0.2))
            self.samples = test_normals + anomalies[:n_test_anomalies]

    def _semantic_to_anomaly_mask(self, sem_img):
        sem = np.array(sem_img)
        mask = np.all(sem == self.ANOMALY_COLOR, axis=-1)
        return Image.fromarray(mask.astype(np.uint8) * 255)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, sem_path, label = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")

        if self.transform:
            rgb = self.transform(rgb)

        if self.mode == "train":
            return rgb

        semantic = Image.open(sem_path)
        semantic = self._semantic_to_anomaly_mask(semantic)

        anomaly_mask = None
        if self.sem_transform:
            anomaly_mask = self.sem_transform(semantic).int()

        return rgb, label, anomaly_mask, str(rgb_path)


if __name__ == "__main__":
    # set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --------------------- COPY FROM HERE -----------------------
    # define torchvision transformations
    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            antialias=True,
        ),
        transforms.ToTensor()
    ])
    sem_transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            antialias=True,
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor()
    ])

    # train and test datasets
    root_dir = "/anovox/Anovox_Sample/Anovox"
    dataset = AnoVoxDataset(  # DELETE THIS LINE (debug only)
        root_dir=root_dir,  # DELETE THIS LINE
        mode="all",  # DELETE THIS LINE
    )  # DELETE THIS LINE
    train_dataset = AnoVoxDataset(
        root_dir=root_dir,
        mode="train",
        normal_split_ratio=0.8,  # default value
        transform=transform,
        sem_transform=sem_transform
    )
    test_dataset = AnoVoxDataset(
        root_dir=root_dir,
        mode="test",
        transform=transform,
        sem_transform=sem_transform
    )
    # --------------------- TO HERE -----------------------

    # print datasets' size
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)} / {len(dataset)}")
    print(f"Test dataset size: {len(test_dataset)} / {len(dataset)}\n")

    # print total anomalous samples
    print(f"Total anomalous samples: {len(dataset) - (len(train_dataset) / 0.8):.0f} / {len(dataset)}\n")

    # print image type
    print(f"Dataset image type: {type(dataset[0][0])}")
    print(f"Dataset transformed image type: {type(train_dataset[0][0])}\n")

    # print image shape
    print(f"Dataset image shape: {dataset[0][0].size}")
    print(f"Dataset transformed image shape: {train_dataset[0][0].shape}\n")