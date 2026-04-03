import random
import argparse
import gc
import os
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from moviad.common.common_utils import obsolete
from moviad.datasets.ad_datasets import AnoVoxDataset
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.trainers.trainer_rd4ad import TrainerRD4AD


def train_rd4ad(backbone: str, save_path: str, device: torch.device, epochs: int = 100, seed: int = 1):

    print(f"Training RD4AD with {backbone} backbone on AnoVox dataset...")

    # define training and test datasets
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
    root_dir = "Anovox"

    train_dataset = AnoVoxDataset(
        root_dir=root_dir,
        mode="train",
        normal_split_ratio=0.8,
        transform=transform,
        sem_transform=sem_transform
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = AnoVoxDataset(
        root_dir=root_dir,
        mode="test",
        transform=transform,
        sem_transform=sem_transform
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}\n")

    # define the model
    model = RD4AD(backbone, device, input_size=(224, 224))
    model.to(device)
    model.train()

    # save the model
    if save_path:
        save_path = os.path.join(save_path, "anovox", "rd4ad")
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{backbone}_seed_{seed}.pth")

    trainer = TrainerRD4AD(model = model,
                           train_dataloader = train_dataloader,
                           eval_dataloader = test_dataloader,
                           device = device,
                           save_path = full_path if save_path else None)
    trainer.train(epochs)

    # force garbage collector in case
    del model
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    #parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        train_rd4ad(args.backbone, args.save_path, device, args.epochs, args.seed)


if __name__ == "__main__":
    main()
