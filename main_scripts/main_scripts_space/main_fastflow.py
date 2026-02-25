import os
import random
import argparse
import gc
import torch

from torchvision import transforms

from moviad.datasets.space_datasets import LunarDataset, MarsDataset, compute_mean_std
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.utilities.evaluator import Evaluator


def train_fastflow(dataset_type: str, backbone: str, save_path: str, device: torch.device, contamination_ratio, test_positive_ratio, epochs: int = 100):
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    # initialize FastFlow model to get the feature extractor
    print(f"Training Fastflow for for {dataset_type} dataset... \n")
    tmp_model = create_fastflow(img_size, backbone, device=device)
    feature_extractor = tmp_model._extract_features

    # define training and test datasets
    if dataset_type == "mars":
        mars_mean, mars_std = compute_mean_std(
            MarsDataset(root_dir="vad_space_datasets/mars", split="train", transform=None))
        mars_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
        ])

        train_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="train",
                                    transform=mars_transform, contamination_ratio=contamination_ratio)

    elif dataset_type == "lunar":
        train_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="train", transform=None, contamination_ratio=contamination_ratio)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(f"Length train dataset: {len(train_dataset)}")

    # define test dataset
    if dataset_type == "mars":
        test_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="test", transform=mars_transform, test_positive_ratio=test_positive_ratio)

    elif dataset_type == "lunar":
        test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None, test_positive_ratio=test_positive_ratio)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    print(f"Length test dataset: {len(test_dataset)}")

    model = create_fastflow(img_size, backbone, device=device).to(device)

    trainer = TrainerFastFlow(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        device=device,
        save_path=save_path if save_path else None
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


def test_fastflow(dataset_type: str, backbone: str, save_path: str, device: torch.device, test_positive_ratio):
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    # initialize FastFlow model to get the feature extractor
    print(f"Training Fastflow for for {dataset_type} dataset... \n")
    tmp_model = create_fastflow(img_size, backbone, device=device)
    feature_extractor = tmp_model._extract_features

    # define training and test datasets
    if dataset_type == "mars":
        mars_mean, mars_std = compute_mean_std(
            MarsDataset(root_dir="vad_space_datasets/mars", split="train", transform=None))
        mars_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
        ])

        test_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="test", transform=mars_transform, test_positive_ratio=test_positive_ratio)

    elif dataset_type == "lunar":
        test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None, test_positive_ratio=test_positive_ratio)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    print(f"Length test dataset: {len(test_dataset)}")

    model = create_fastflow(img_size, backbone, device=device).to(device)
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    evaluator = Evaluator(test_dataloader, device)
    results = evaluator.evaluate_vad_space(model)

    print("Evaluation performances:")
    print(f"""
            img_roc: {results["img_roc_auc"]} \n
            pxl_roc: {results["pxl_roc_auc"]} \n
            f1_img: {results["img_f1"]} \n
            f1_pxl: {results["pxl_f1"]} \n
            img_pr: {results["img_pr_auc"]} \n
            pxl_pr: {results["pxl_pr_auc"]} \n
            pxl_pro: {results["pxl_au_pro"]} \n
        """)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--contamination", type=float, default=None, help="Contamination ratio")
    parser.add_argument("--test_positive_ratio", type=float, default=None, help="Positive ratio")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        train_fastflow(args.dataset_type, args.backbone, args.save_path, device, args.contamination, args.test_positive_ratio, args.epochs)
    elif args.mode == "test":
        test_fastflow(args.dataset_type, args.backbone, args.save_path, device, args.test_positive_ratio)


if __name__ == "__main__":
    main()
