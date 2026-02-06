import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.utilities.configurations import TaskType, Split


def train_fastflow(dataset_path: str, category: str, backbone: str, save_path: str,
                    device: torch.device, epochs: int = 100, max_dataset_size: int = None):

    print(f"Training Fastflow for category: {category} \n")

    # define training and test datasets
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train")
    train_dataset.load_dataset()
    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test")
    test_dataset.load_dataset()
    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # define the model
    model = create_fastflow((224,224), backbone, device)

    trainer = TrainerFastFlow(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
    )
    trainer.train(epochs)

    # save the model
    if save_path:
        torch.save(model.state_dict(), save_path)

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
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
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
        train_fastflow(args.dataset_path, args.category, args.backbone, args.save_path, device, args.epochs)


if __name__ == "__main__":
    main()
