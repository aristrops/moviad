import os
import random
import argparse
import gc
import pandas as pd
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from moviad.datasets.ad_datasets import AnoVoxDataset
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.utilities.evaluator import Evaluator

from moviad.models.patchcore.autoencoder import FeatureAutoencoder

def append_results_to_csv(csv_path: str, row: dict):
    df_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(csv_path, index=False)

def train_fastflow(backbone: str, save_path: str, device: torch.device, epochs: int = 100, seed = 1):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    print(f"Training Fastflow with {backbone} backbone on AnoVox dataset...")

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

    model = create_fastflow(img_size, backbone, device=device).to(device)

    # save the model
    if save_path:
        save_path = os.path.join(save_path, "anovox", "fastflow")
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{backbone}_seed_{seed}.pth")

    trainer = TrainerFastFlow(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        device=device,
        save_path=full_path if save_path else None
    )
    trainer.train(epochs)

    # force garbage collector in case
    del model
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()



def test_fastflow(backbone: str, save_path: str, device: torch.device):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    print(f"Evaluating Fastflow with {backbone} backbone on AnoVox dataset...")

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

    test_dataset = AnoVoxDataset(
        root_dir=root_dir,
        mode="test",
        transform=transform,
        sem_transform=sem_transform
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    print(f"Test dataset size: {len(test_dataset)}\n")

    #load the model state dict
    save_path = os.path.join(save_path, "anovox", "fastflow")
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, f"{backbone}.pth")
    
    model = create_fastflow(img_size, backbone, device=device).to(device)
    model.eval()
    state_dict = torch.load(full_path, map_location=device)
    model.load_state_dict(state_dict)

    evaluator = Evaluator(test_dataloader, device)
    results = evaluator.evaluate(model)

    # #save results to csv
    # csv_path = "full_results_iot.csv"
    #
    # row = {"method": "fastflow",
    #         "category": category,
    #         "backbone": backbone,
    #         "compress_images": compress_images,
    #         "quality": quality,
    #         "feature_compression_method": feature_compression_method,
    #         "sampling_ratio": sampling_ratio,
    #         "img_roc": results["img_roc_auc"],
    #         "pxl_roc": results["pxl_roc_auc"],
    #         "f1_img": results["img_f1"],
    #         "f1_pxl": results["pxl_f1"],}
    #
    # append_results_to_csv(csv_path, row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
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
        train_fastflow(args.backbone, args.save_path, device, args.epochs, args.seed)
    if args.mode == "test":
        test_fastflow(args.backbone, args.save_path, device)


if __name__ == "__main__":
    main()
