import random
import argparse
import gc

import torch
import torch.nn as nn

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.models.patchcore.product_quantizer import ProductQuantizer

from moviad.models.patchcore.autoencoder import FeatureAutoencoder


def train_patchcore(dataset_path: str, category: str, backbone: str, ad_layers: list,
                    save_path: str, device: torch.device,
                    compress_images: bool, quality: int, feature_compression_method: str, sampling_ratio: int, quantize_mb: bool = False): #IoT scenario params

    # initialize the feature extractor and quantizer
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    feature_quantizer = ProductQuantizer()

    # initize autoencoders
    with torch.no_grad():
        input_dummy = torch.randn((1, 3, 224, 224))
        features_dummy = feature_extractor(input_dummy.to(device))

    autoencoders = nn.ModuleList()
    for layer_features in features_dummy:
        autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
        autoencoders.append(autoencoder)

    optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor
    compressor = CustomFeatureCompressor(device, feature_compression_method=feature_compression_method,
                                         quality=quality, compression_ratio=sampling_ratio, quantizer=feature_quantizer,
                                         autoencoders=autoencoders)

    print(f"Training Pathcore for category: {category} \n")

    # define training dataset
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train",
                                 compressor=compressor, apply_compression=compress_images, quality=quality)
    train_dataset.load_dataset()

    # train compressors and compress features
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
    print(f"Length train dataset: {len(train_dataset)}")

    # define test dataset
    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test",
                                compressor=compressor, apply_compression=compress_images, quality=quality)
    test_dataset.load_dataset()

    # compress features
    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=test_dataset.collate_fn)

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
    print(f"Size of the feature extractor: {sizes['feature_extractor']['size']: .2f} MB")
    print(f"Total model size: {total_size: .2f} MB")

    # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def test_patchcore(dataset_path: str, category: str, backbone: str, ad_layers: list,
                    model_checkpoint_path: str, device: torch.device, compress_images: bool,
                    quality: int, feature_compression_method: str, sampling_ratio: int, quantize_mb: bool = False):

    # initialize the feature extractor and compressor
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    feature_quantizer = ProductQuantizer()

    # initialize autoencoders
    with torch.no_grad():
        input_dummy = torch.randn((1, 3, 224, 224))
        features_dummy = feature_extractor(input_dummy.to(device))

    autoencoders = nn.ModuleList()
    for layer_features in features_dummy:
        autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
        autoencoders.append(autoencoder)

    optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor
    compressor = CustomFeatureCompressor(device, feature_compression_method=feature_compression_method,
                                         quality=quality, compression_ratio=sampling_ratio, quantizer=feature_quantizer,
                                         autoencoders=autoencoders)

    print(f"Testing Pathcore for category: {category} \n")

    if "pq" in feature_compression_method or "ae" in feature_compression_method:
         # define training dataset
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train", compressor=compressor,
                                        apply_compression=compress_images, quality=quality)

        train_dataset.load_dataset()

        # train the compressor on training set
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

    # define test dataset
    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test", compressor=compressor,
                                    apply_compression=compress_images, quality=quality)
    test_dataset.load_dataset()

    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    print(f"Length test dataset: {len(test_dataset)}")


    # define and load the model
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor,
                          compression_method=feature_compression_method, apply_quantization=quantize_mb)
    patchcore.load_model(model_checkpoint_path)
    patchcore.to(device)
    patchcore.eval()

    results = Evaluator.evaluate(patchcore, test_dataloader, device)
    print("Evaluation performances:")
    print(f"""
        img_roc: {results['img_roc_auc']}
        pxl_roc: {results['pxl_roc_auc']}
        f1_img: {results['img_f1']}
        f1_pxl: {results['pxl_f1']}
        img_pr: {results['img_pr_auc']}
        pxl_pr: {results['pxl_pr_auc']}
        pxl_pro: {results['pxl_au_pro']}
        """)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--compress_images", action="store_true", help="Compress images using JPEG or WEBP")
    parser.add_argument("--quality", type=int, default=50, help="Compression quality of images")
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+",
                        help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=0.25,
                        help="Sampling ratio for random projection of features")
    parser.add_argument("--quantize_mb", action="store_true",
                        help="Whether to quantize the memory bank to reduce its size")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_patchcore(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device,
                        args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio, args.quantize_mb)
    elif args.mode == "test":
        test_patchcore(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device,
                       args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio, args.quantize_mb)


if __name__ == "__main__":
    main()