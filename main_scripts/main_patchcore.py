import random
import argparse
import gc
import pathlib
import time
import wandb
import pandas as pd
import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.models.patchcore.product_quantizer import ProductQuantizer

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


def train_patchcore(dataset_type: str, dataset_path: str, categories: str, backbone: str, ad_layers: list, compress_images,
                    quality, image_compression_method,
                    feature_compression_method, sampling_ratio, pq_method, pq_subspaces, save_path: str,
                    device: torch.device, max_dataset_size: int = None, quantize_mb: bool = False):
    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    feature_quantizer = ProductQuantizer(subspaces=pq_subspaces)
    compressor = CustomFeatureCompressor(device, image_compression_method=image_compression_method,
                                         feature_compression_method=feature_compression_method,
                                         quality=quality, compression_ratio=sampling_ratio, pq_method=pq_method,
                                         quantizer=feature_quantizer)

    for category in categories:

        print(f"Training Pathcore for category: {category} \n")

        # define training and test datasets

        if dataset_type == "mvtec":
            train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train", compressor=compressor,
                                        apply_compression=compress_images, quality=quality,
                                        compression_method=image_compression_method)
        elif dataset_type == "visa":
            train_dataset = VisaDataset(dataset_path, csv_path=os.path.join(dataset_path, "split_csv", "1cls.csv"),
                                       split=Split.TRAIN, class_name=category)
        
        train_dataset.load_dataset()
    
        if max_dataset_size is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
        if feature_compression_method is not None:
            if "pq" in feature_compression_method:
                feature_vectors = compressor.collect_feature_vectors(train_dataset, feature_extractor)
                compressor.fit_quantizers(feature_vectors)
            train_dataset = CompressedFeaturesDataset(feature_extractor, train_dataset, compressor, device)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                                           collate_fn=train_dataset.collate_fn)
        else:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        print(f"Length train dataset: {len(train_dataset)}")
        
        if dataset_type == "mvtec":
            test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test", compressor=compressor,
                                        apply_compression=compress_images, quality=quality,
                                        compression_method=image_compression_method)
        elif dataset_type == "visa":
            test_dataset = VisaDataset(dataset_path, csv_path=os.path.join(dataset_path, "split_csv", "1cls.csv"),
                                       split=Split.TEST, class_name=category)
        test_dataset.load_dataset()

        if max_dataset_size is not None:
            test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
        if feature_compression_method is not None:
            if "pq" in feature_compression_method:
                feature_vectors = compressor.collect_feature_vectors(test_dataset, feature_extractor)
                compressor.fit_quantizers(feature_vectors)
            test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True,
                                                          collate_fn=test_dataset.collate_fn)
        else:
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
        print(f"Length test dataset: {len(test_dataset)}")

        # define the model
        patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor,
                              compression_method=feature_compression_method, apply_quantization=quantize_mb)
        patchcore.to(device)
        patchcore.train()

        trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device)
        results = trainer.train()

        # save the model
        if save_path:
            patchcore.save_model(save_path)

        sizes, total_size = patchcore.get_model_size_and_macs()
        print(f"Size of the memory bank: {sizes['memory_bank']['size']: .2f} MB")
        print(f"Total model size: {total_size: .2f} MB")

        # save results to csv
        csv_path = "quantized_patchcore_visa.csv"

        row = {"category": category,
               "ad_layers": encode_ad_layers(ad_layers),
               "quantize_mb": quantize_mb,
               "img_roc": results["img_roc"],
               "pxl_roc": results["pxl_roc"],
               "f1_img": results["f1_img"],
               "f1_pxl": results["f1_pxl"],
               "img_pr": results["img_pr"],
               "pxl_pr": results["pxl_pr"],
               "pxl_pro": results["pxl_pro"],
               "memory_bank_size_MB": sizes["memory_bank"]["size"],}
        
        append_results_to_csv(csv_path, row)


        # force garbage collector in case
        del patchcore
        del test_dataset
        del train_dataset
        del train_dataloader
        del test_dataloader
        torch.cuda.empty_cache()
        gc.collect()


def test_patchcore(dataset_type: str, dataset_path: str, categories: str, backbone: str, ad_layers: list, model_checkpoint_path, compress_images,
                    quality, image_compression_method, visual_test_path,
                    feature_compression_method, sampling_ratio, pq_method, pq_subspaces,
                    device: torch.device, max_dataset_size: int = None, quantize_mb: bool = False):

    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    feature_quantizer = ProductQuantizer(subspaces=pq_subspaces)
    compressor = CustomFeatureCompressor(device, image_compression_method=image_compression_method,
                                         feature_compression_method=feature_compression_method,
                                         quality=quality, compression_ratio=sampling_ratio, pq_method=pq_method,
                                         quantizer=feature_quantizer)

    for category in categories:
        if dataset_type == "mvtec":
            test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test", compressor=compressor,
                                    apply_compression=compress_images, quality=quality,
                                    compression_method=image_compression_method)
        elif dataset_type == "visa":
            test_dataset = VisaDataset(dataset_path, csv_path=os.path.join(dataset_path, "split_csv", "1cls.csv"),
                                       split=Split.TEST, class_name=category)

        test_dataset.load_dataset()

        if max_dataset_size is not None:
            test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
        if feature_compression_method is not None:
            if "pq" in feature_compression_method:
                feature_vectors = compressor.collect_feature_vectors(test_dataset, feature_extractor)
                compressor.fit_quantizers(feature_vectors)
            test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True,
                                                          collate_fn=test_dataset.collaimte_fn)
        else:
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        print(f"Length test dataset: {len(test_dataset)}")

        # load the model
        patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor,
                              compression_method=feature_compression_method, apply_quantization=quantize_mb)
        patchcore.load_model(model_checkpoint_path)
        patchcore.to(device)
        patchcore.eval()

        evaluator = Evaluator(test_dataloader, device)
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore)

        print("Evaluation performances:")
        print(f"""
        img_roc: {img_roc}
        pxl_roc: {pxl_roc}
        f1_img: {f1_img}
        f1_pxl: {f1_pxl}
        img_pr: {img_pr}
        pxl_pr: {pxl_pr}
        pxl_pro: {pxl_pro}
        """)

        # chek for the visual test
        if visual_test_path:

            # Get output directory.
            dirpath = pathlib.Path(visual_test_path)
            dirpath.mkdir(parents=True, exist_ok=True)

            for images, labels, masks, paths in tqdm(iter(test_dataloader)):
                anomaly_maps, pred_scores = patchcore(images.to(device))

                anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

                for i in range(anomaly_maps.shape[0]):
                    patchcore.save_anomaly_map(visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                               labels[i], masks[i])


def main():
    categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut",
                  "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_type", type=str, choices=["mvtec", "visa"], help="Type of dataset to use")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--categories", type=str, nargs="+", default=categories, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--compress_images", action="store_true", help="Compress images using JPEG or WEBP")
    parser.add_argument("--quality", type=int, default=50, help="Compression quality of images")
    parser.add_argument("--image_compression_method", type=str, default="WEBP", help="Method for image compression")
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+",
                        help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=0.25,
                        help="Sampling ratio for random projection of features")
    parser.add_argument("--pq_method", type=str, default=None, help="Which PQ method to use")
    parser.add_argument("--pq_subspaces", type=int, default=None, help="PQ subspaces to use")
    parser.add_argument("--quantize_mb", action="store_true", help="Whether to quantize the memory bank to reduce its size")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_patchcore(args.dataset_type, args.dataset_path, args.categories, args.backbone, args.ad_layers,
                        args.compress_images, args.quality, args.image_compression_method,
                        args.feature_compression_method, args.sampling_ratio, args.pq_method, args.pq_subspaces,
                        args.save_path, device, quantize_mb = args.quantize_mb)
    elif args.mode == "test":
        test_patchcore(args.dataset_type, args.dataset_path, args.categories, args.backbone, args.ad_layers, args.save_path, args.compress_images,
                       args.quality, args.image_compression_method,  args.visual_test_path, args.feature_compression_method,
                       args.sampling_ratio, args.pq_method, args.pq_subspaces, device, quantize_mb = args.quantize_mb)


if __name__ == "__main__":
    main()