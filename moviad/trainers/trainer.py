import torch

from moviad.datasets.iad_dataset import IadDataset


class Trainer:
    train_dataloader: IadDataset
    test_dataloader: IadDataset
    model: torch.nn.Module
    device: torch.device