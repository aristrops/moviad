import random
import argparse
import gc
import pathlib
import pandas as pd
import os

import torch
from tqdm import tqdm
from torchvision import transforms

from moviad.datasets.space_datasets import LunarDataset, MarsDataset, compute_mean_std
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.evaluator import Evaluator
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor

AD_LAYERS = {
    ("features.4", "features.7", "features.10"): "low",
    ("features.7", "features.10", "features.13"): "mid",
    ("features.10", "features.13", "features.16"): "high",
    ("features.3", "features.8", "features.14"): "equiv"
}


def encode_ad_layers(ad_layers):
    key = tuple(ad_layers)

    if key in AD_LAYERS:
        return AD_LAYERS[key]

    return "custom:" + ",".join(key)


def append_results_to_csv(csv_path: str, row: dict):
    df_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(csv_path, index=False)


def train_patchcore(dataset_type: str, backbone: str, ad_layers: list, save_path: str,
                    device: torch.device, contamination_ratio: float, test_positive_ratio: float):

    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)

    print(f"Training Pathcore for {dataset_type} dataset...\n")

    #define training dataset
    if dataset_type == "mars":
        mars_mean, mars_std = compute_mean_std(MarsDataset(root_dir=r"vad_space_datasets/mars", split="train", transform=None))
        mars_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
        ])

        train_dataset = MarsDataset(root_dir=r"vad_space_datasets/mars", split="train", transform=mars_transform, contamination_ratio=contamination_ratio)

    elif dataset_type == "lunar":
        train_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="train", transform=None, contamination_ratio=contamination_ratio)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(f"Length train dataset: {len(train_dataset)}")

    # define test dataset
    if dataset_type == "mars":
        test_dataset = MarsDataset(root_dir=r"vad_space_datasets/mars", split="test", transform=mars_transform, test_positive_ratio=test_positive_ratio)
    elif dataset_type == "lunar":
        test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None, test_positive_ratio=test_positive_ratio)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    print(f"Length test dataset: {len(test_dataset)}")

    # define the model
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor,
                          compression_method=None, apply_quantization=False)
    patchcore.to(device)
    patchcore.train()

    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device)
    results = trainer.train()

    # save the model
    if save_path:
        patchcore.save_model(save_path)

    # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def test_patchcore(dataset_type: str, backbone: str, ad_layers: list, model_checkpoint_path,
                   device: torch.device, test_positive_ratio: float):

    print(f"Testing Pathcore for {dataset_type} dataset...\n")

    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)

    if dataset_type == "mars":
        mars_mean, mars_std = compute_mean_std(MarsDataset(root_dir=r"vad_space_datasets/mars", split="train", transform=None))
        mars_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
        ])
        test_dataset = MarsDataset(root_dir=r"vad_space_datasets/mars", split="test", transform=mars_transform,
                                   test_positive_ratio=test_positive_ratio)
    elif dataset_type == "lunar":
        test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None,
                                    test_positive_ratio=test_positive_ratio)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    print(f"Length test dataset: {len(test_dataset)}")

    # load the model
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor,
                          compression_method=None, apply_quantization=False)
    patchcore.load_model(model_checkpoint_path)
    patchcore.to(device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, device)
    results = evaluator.evaluate_vad_space(patchcore)

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
    categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut",
                  "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_type", type=str, help="Type of dataset to use")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--contamination", type=float, default=None, help="Contamination ratio")
    parser.add_argument("--test_positive_ratio", type=float, default=None, help="Positive ratio")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_patchcore(args.dataset_type, args.backbone, args.ad_layers, args.save_path, device, args.contamination,args.test_positive_ratio)
    elif args.mode == "test":
        test_patchcore(args.dataset_type, args.backbone, args.ad_layers, args.save_path, device, args.test_positive_ratio)


if __name__ == "__main__":
    main()