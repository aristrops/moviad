import random
import argparse
import gc

import torch
import torch.nn as nn

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.utilities.configurations import TaskType
from moviad.utilities.evaluator import Evaluator
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.utilities.feature_compressor import CustomFeatureCompressor
from moviad.models.patchcore.product_quantizer import ProductQuantizer

from moviad.utilities.autoencoder import FeatureAutoencoder


def train_fastflow(dataset_path: str, category: str, backbone: str, ad_layers: list, save_path: str, device: torch.device,
                   compress_images: bool, quality: int, feature_compression_method: str, sampling_ratio: int, #IoT scenario params
                   epochs: int = 100):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    # initialize FastFlow model to get the feature extractor
    tmp_model = create_fastflow(img_size, backbone, ad_layers, feature_compression_method, device=device)
    feature_extractor = tmp_model._extract_features

    # initialize autoencoders
    with torch.no_grad():
        input_dummy = torch.randn((1, 3, 224, 224))
        features_dummy = feature_extractor(input_dummy.to(device))

    autoencoders = nn.ModuleList()
    for layer_features in features_dummy:
        autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
        autoencoders.append(autoencoder)

    optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor and quantizer
    feature_quantizer = ProductQuantizer()
    compressor = CustomFeatureCompressor(device,feature_compression_method=feature_compression_method, quality=quality,
                                         compression_ratio=sampling_ratio, quantizer=feature_quantizer, img_size=img_size, autoencoders=autoencoders)

    print(f"Training Fastflow for category: {category} \n")

    # define training dataset
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train",
                                 compressor = compressor, apply_image_compression = compress_images, quality = quality, img_size=img_size)
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

    #define test dataset
    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test",
                                compressor = compressor, apply_image_compression = compress_images, quality = quality, img_size=img_size)
    test_dataset.load_dataset()

    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split = "test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)

    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Length test dataset: {len(test_dataset)}")

    # define the model
    model = create_fastflow(img_size, backbone, ad_layers, feature_compression_method, sampling_ratio, device=device).to(device)

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


def test_fastflow(dataset_path: str, category: str, backbone: str, ad_layers: list, save_path: str, device: torch.device,
                  compress_images: bool, quality: int, feature_compression_method: str, sampling_ratio: int):
    
    if backbone == "cait_m48_448":
        img_size = (448, 448)
    else:
        img_size = (224, 224)

    # initialize FastFlow model to get the feature extractor
    tmp_model = create_fastflow(img_size, backbone, ad_layers, feature_compression_method, device=device)
    feature_extractor = tmp_model._extract_features

    # initialize autoencoders
    with torch.no_grad():
        input_dummy = torch.randn((1, 3, 224, 224))
        features_dummy = feature_extractor(input_dummy.to(device))

    autoencoders = nn.ModuleList()
    for layer_features in features_dummy:
        autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
        autoencoders.append(autoencoder)

    optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor and quantizer
    feature_quantizer = ProductQuantizer()
    compressor = CustomFeatureCompressor(device,feature_compression_method=feature_compression_method, quality=quality,
                                         compression_ratio=sampling_ratio, quantizer=feature_quantizer, img_size=img_size, autoencoders=autoencoders)

    print(f"Testing Fastflow for category: {category} \n")

    if "pq" in feature_compression_method or "ae" in feature_compression_method:
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train",
                                     compressor=compressor, apply_image_compression=compress_images, quality=quality,
                                     img_size=img_size)
        train_dataset.load_dataset()

        # train compressor on the training set
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
    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test",
                                compressor=compressor, apply_image_compression=compress_images, quality=quality,
                                img_size=img_size)
    test_dataset.load_dataset()

    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split = "test")   
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Length test dataset: {len(test_dataset)}")

    # define and load the model
    model = create_fastflow(img_size, backbone, ad_layers, feature_compression_method, sampling_ratio, device=device).to(device)
    model.eval()
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)

    evaluator = Evaluator(test_dataloader, device)
    results = evaluator.evaluate(model)
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
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+", help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Sampling ratio for random projection of features")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        train_fastflow(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device,
                       args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio, args.epochs)
    if args.mode == "test":
        test_fastflow(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device,
                      args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio)

if __name__ == "__main__":
    main()
