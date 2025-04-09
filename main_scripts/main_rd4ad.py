import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.trainers.trainer_rd4ad import TrainerRD4AD
from moviad.utilities.configurations import TaskType, Split


def train_rd4ad(dataset_path: str, category: str, backbone: str, ad_layers: list, save_path: str,
                    device: torch.device, epochs: int = 100, max_dataset_size: int = None):

    print(f"Training RD4AD for category: {category} \n")

    # define training and test datasets
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train")
    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test")
    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # define the model
    model = RD4AD(backbone, device, input_size=(224, 224))
    model.to(device)
    model.train()

    trainer = TrainerRD4AD(model, train_dataloader, test_dataloader, device)
    trainer.train(epochs)

    # save the model
    if save_path:
        torch.save(patchcore.state_dict(), save_path)

        # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
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
        train_rd4ad(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device, args.epochs)


if __name__ == "__main__":
    main()
