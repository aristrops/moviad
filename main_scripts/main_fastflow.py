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
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.models.patchcore.product_quantizer import ProductQuantizer


def train_fastflow(dataset_path: str, category: str, backbone: str, compress_images, quality,
                   feature_compression_method, sampling_ratio, pq_subspaces, 
                   save_path: str, device: torch.device, epochs: int = 100, max_dataset_size: int = None):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)
    
    #initialize feature compressor and quantizer
    feature_quantizer = ProductQuantizer(subspaces=pq_subspaces)
    compressor = CustomFeatureCompressor(device,feature_compression_method=feature_compression_method, quality=quality,
                                         compression_ratio=sampling_ratio, quantizer=feature_quantizer, img_size=img_size)
    
    #initialize FastFlow model to get the feature extractor
    print(f"Training Fastflow for category: {category} \n")
    tmp_model = create_fastflow(img_size, backbone, feature_compression_method, device=device)
    feature_extractor = tmp_model._extract_features

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
        
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device)   
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = create_fastflow(img_size, backbone, feature_compression_method, sampling_ratio, device=device).to(device)

    trainer = TrainerFastFlow(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
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

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        train_fastflow(args.dataset_path, args.category, args.backbone, args.compress_images, args.quality, args.feature_compression_method, 
                       args.sampling_ratio, args.pq_subspaces,args.save_path, device, args.epochs)


if __name__ == "__main__":
    main()
