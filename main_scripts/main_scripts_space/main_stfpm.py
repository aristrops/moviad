import os, sys, json, logging
from datetime import datetime
import argparse
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, "..", ".."))

from moviad.models.stfpm.stfpm import Stfpm
from moviad.trainers.trainer_stfpm import train_model
from moviad.datasets.space_datasets import LunarDataset, MarsDataset, compute_mean_std
from moviad.utilities.evaluator import Evaluator


# 1. train stfpm on a category
# 2. save the best model
# 3. load the model (if not trained in the same run)
# 4. run test on the model
# 5. compute the metrics


def main(args):
    dataset_type = args.dataset_type
    model_name = args.model_name
    ad_layers = args.ad_layers
    seeds = args.seeds
    epochs = args.epochs
    save_path = args.save_path
    contamination = args.contamination
    test_positive_ratio = args.test_positive_ratio

    device = torch.device(args.device)
    input_sizes = {
        "mcunet-in3": (176, 176),
        "mobilenet_v2": (224, 224),
        "phinet_1.2_0.5_6_downsampling": (224, 224),
        "wide_resnet50_2": (224, 224),
        "micronet-m1": (224, 224),
    }

    if args.input_size is not None:
        input_sizes[model_name] = args.input_size

    # TRAIN AND SAVE BEST ###############################################
    if args.train:
        ad_model = "stfpm"

        #ad_layers = [ad_layers]

        print(f"Training STFPM for {dataset_type} dataset... \n")

        if dataset_type == "mars":
            mars_mean, mars_std = compute_mean_std(
                MarsDataset(root_dir=r"vad_space_datasets/mars", split="train", transform=None))
            mars_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
            ])

            train_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="train",
                                        transform=mars_transform, contamination_ratio=contamination)

        elif dataset_type == "lunar":
            train_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="train", transform=None, contamination_ratio=contamination)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        print(f"Length train dataset: {len(train_dataset)}")

        if dataset_type == "mars":
            test_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="test",
                                       transform=mars_transform, test_positive_ratio=test_positive_ratio)

        elif dataset_type == "lunar":
            test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None, test_positive_ratio=test_positive_ratio)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
        print(f"Length test dataset: {len(test_dataset)}")

        model = Stfpm(backbone_model_name=model_name, ad_layers=ad_layers)
        model.to(device)

        train_model(model, train_dataloader, test_dataloader, epochs, device, model_save_path=save_path, seed=seeds)

    if args.eval:
        print(f"Evaluating STFPM for {dataset_type} dataset... \n")

        if dataset_type == "mars":
            mars_mean, mars_std = compute_mean_std(
                MarsDataset(root_dir="vad_space_datasets/mars", split="train", transform=None))
            mars_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
            ])

            test_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="test", transform=mars_transform,
                                       test_positive_ratio=test_positive_ratio)

        elif dataset_type == "lunar":
            test_dataset = LunarDataset(root_dir="vad_space_datasets/lunar", split="test", transform=None,
                                        test_positive_ratio=test_positive_ratio)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
        print(f"Length test dataset: {len(test_dataset)}")

        model = Stfpm(backbone_model_name=model_name, ad_layers=ad_layers)
        model.to(device)

        print(f"Loading model from {save_path}")
        checkpoint = torch.load(save_path, map_location=device)

        # Case 1: checkpoint contains model_state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Case 2: checkpoint IS the state_dict
            model.load_state_dict(checkpoint)

        evaluator = Evaluator(
            test_dataloader=test_dataloader,
            device=device
        )

        metrics = evaluator.evaluate_vad_space(model)

        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")



if __name__ == "__main__":
    # try:
    #     paths = json.load(open("../../utilities/pathconfig.json"))
    # except FileNotFoundError:
    #     logging.warning("Could not load pathconfig.json. Make sure the file exists.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--ad_layers", type=int, nargs="+")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--output_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--input_size", type=int, nargs=2, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--disable_dataset_norm", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--feature_maps_dir", type=str, default=None)
    parser.add_argument("--contamination", type=float, default=None, help="Contamination ratio")
    parser.add_argument("--test_positive_ratio", type=float, default=None, help="Positive ratio")

    args = parser.parse_args()

    main(args)