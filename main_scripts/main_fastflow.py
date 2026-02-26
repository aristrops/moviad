import os
import random
import argparse
import gc
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.utilities.configurations import TaskType, Split
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.models.patchcore.product_quantizer import ProductQuantizer

from moviad.models.patchcore.autoencoder import FeatureAutoencoder


def train_fastflow(dataset_path: str, category: str, backbone: str, compress_images, quality,
                   feature_compression_method, sampling_ratio, pq_subspaces, 
                   save_path: str, device: torch.device, epochs: int = 100, max_dataset_size: int = None):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    #initialize FastFlow model to get the feature extractor
    print(f"Training Fastflow for category: {category} \n")
    tmp_model = create_fastflow(img_size, backbone, feature_compression_method, device=device)
    feature_extractor = tmp_model._extract_features

    with torch.no_grad():
        input_dummy = torch.randn((1, 3, 224, 224))
        features_dummy = feature_extractor(input_dummy.to(device))

    autoencoders = nn.ModuleList()
    for layer_features in features_dummy:
        autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
        autoencoders.append(autoencoder)

    optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    #initialize feature compressor and quantizer
    feature_quantizer = ProductQuantizer(subspaces=pq_subspaces)
    compressor = CustomFeatureCompressor(device,feature_compression_method=feature_compression_method, quality=quality,
                                         compression_ratio=sampling_ratio, quantizer=feature_quantizer, img_size=img_size, autoencoders=autoencoders)

    # define training and test datasets
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train", compressor, compress_images, quality, img_size=img_size)
    train_dataset.load_dataset()
    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")

    if feature_compression_method is not None:
        if "pq" in feature_compression_method:
            feature_vectors = compressor.collect_feature_vectors(train_dataset, feature_extractor)
            compressor.fit_quantizers(feature_vectors)
        if "ae" in feature_compression_method:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            compressor.train_autoencoders(
                train_dataloader=train_dataloader,
                feature_extractor=feature_extractor,
                optimizers=optimizers,
                device=device,
                epochs=10,
                noise_std=0.001,
            )

        train_dataset = CompressedFeaturesDataset(feature_extractor, train_dataset, compressor, device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test", compressor, compress_images, quality, img_size=img_size)
    test_dataset.load_dataset()
    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")

    if feature_compression_method is not None:
        if "pq" in feature_compression_method:
            feature_vectors = compressor.collect_feature_vectors(test_dataset, feature_extractor)
            compressor.fit_quantizers(feature_vectors)
        if "ae" in feature_compression_method:
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
            compressor.test_reconstruction(  # Only to check overfitting, in real cases test is likely not feasible
                test_dataloader=test_dataloader,
                feature_extractor=feature_extractor,
                device=device,
            )
        
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split = "test")   
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = create_fastflow(img_size, backbone, feature_compression_method, sampling_ratio, device=device).to(device)

    # save the model
    if save_path:
        save_path = os.path.join(save_path, "fastflow", category)
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{backbone}_compress_images_{compress_images}_feature_compression_{feature_compression_method}_sampling_{sampling_ratio}.pt")

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--categories", type=str, nargs ="+", help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--compress_images", action="store_true", help="Compress images using JPEG or WEBP")
    parser.add_argument("--quality", type=int, default=50, help="Compression quality of images")
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+", help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Sampling ratio for random projection of features")
    parser.add_argument("--pq_subspaces", type=int, default=None, help="PQ subspaces to use")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut",
                  "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        for category in categories:
            train_fastflow(args.dataset_path, category, args.backbone, args.compress_images, args.quality, args.feature_compression_method, 
                       args.sampling_ratio, args.pq_subspaces,args.save_path, device, args.epochs)


if __name__ == "__main__":
    main()
