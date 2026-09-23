import time
import gc
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import random
import os

from moviad.dinomaly.models.uad import ViTill
from moviad.dinomaly.models import vit_encoder
from moviad.dinomaly.dinov1.utils import trunc_normal_
from moviad.dinomaly.models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.dinomaly.utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler
from functools import partial
from moviad.dinomaly.optimizers import StableAdamW
import warnings
import logging

from moviad.utilities.configurations import TaskType
from moviad.models.patchcore.product_quantizer import ProductQuantizer
from moviad.models.patchcore.features_dataset import CompressedFeaturesDataset
from moviad.models.patchcore.autoencoder import FeatureAutoencoder


warnings.filterwarnings("ignore")


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# function to set up the model
def create_dinomaly(encoder_name: str, device: torch.device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    encoder.to(device)

    if 'tiny' in encoder_name:
        embed_dim, num_heads = 192, 3
    elif 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in tiny, small, base, large.")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                 attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    trainable = nn.ModuleList([bottleneck, decoder])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    return model, trainable


def train_dinomaly(dataset_path: str, category: str, encoder_name: str, save_path: str, device: torch.device,
                   compress_images: bool, quality: int, feature_compression_method: str, sampling_ratio: int,
                   epochs: int = 100, seed: int = 2):
    setup_seed(seed)
    batch_size = 16
    image_size = (448, 448)

    # initialize model to get the feature extractor
    feature_extractor = vit_encoder.load(encoder_name)
    feature_extractor.to(device)

    # initialize autoencoders
    autoencoders = None
    optimizers = None
    if feature_compression_method is not None and "ae" in feature_compression_method:
        with torch.no_grad():
            input_dummy = torch.randn((1, 3, 224, 224))
            features_dummy = feature_extractor(input_dummy.to(device))

        autoencoders = nn.ModuleList()
        for layer_features in features_dummy:
            autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
            autoencoders.append(autoencoder)

        optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor
    feature_quantizer = ProductQuantizer(subspaces=None)
    compressor = CustomFeatureCompressor(device, feature_compression_method=feature_compression_method, quality=quality,
                                         compression_ratio=sampling_ratio, quantizer=feature_quantizer, img_size=image_size, autoencoders=autoencoders)

    print(f"Training Dinomaly for category: {category} \n")

    # define training dataset
    train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train",
                                 compressor=compressor, apply_compression=compress_images, quality=quality, img_size=image_size)
    train_dataset.load_dataset()

    # train compressor and compress features
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
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                                       collate_fn=train_dataset.collate_fn)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f"Length train dataset: {len(train_dataset)}")

    # define test dataset
    test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test",
                                compressor=compressor, apply_compression=compress_images, quality=quality, img_size=image_size)
    test_dataset.load_dataset()

    # compress features
    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                                      collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Length test dataset: {len(test_dataset)}")

    # define the model
    model, trainable = create_dinomaly(encoder_name, device)
    model.train()

    total_iters = int(np.ceil(epochs * len(train_dataloader) / batch_size))

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print(f"Train image number:{len(train_dataset)}")

    it = 0
    epoch_count = 0
    best_auroc = -float("inf")

    for epoch in range(epochs):
        model.train()
        start_epoch_time = time.time()
        loss_list = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)

        for img in progress_bar:
            img = img.to(device)
            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()

            loss_list.append(loss.item())
            lr_scheduler.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} finished in {epoch_time / 60:.2f} minutes")
        epoch_count += 1

        if (epoch_count + 1) % 10 == 0:
            print("Evaluating model...")
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            print(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    category, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

            if auroc_sp > best_auroc and save_path:
                best_auroc = auroc_sp
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved at {save_path}")

            model.train()

    # force garbage collector in case
    del model
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def test_dinomaly(dataset_path: str, category: str, backbone: str, save_path: str, device: torch.device,
                  compress_images: bool, quality: int, feature_compression_method: str, sampling_ratio: int):

    image_size = (448, 448)

    # initialize model to get the feature extractor
    feature_extractor = vit_encoder.load(backbone)
    feature_extractor.to(device)

    # initialize autoencoders
    autoencoders = None
    optimizers = None
    if feature_compression_method is not None and "ae" in feature_compression_method:
        with torch.no_grad():
            input_dummy = torch.randn((1, 3, 224, 224))
            features_dummy = feature_extractor(input_dummy.to(device))

        autoencoders = nn.ModuleList()
        for layer_features in features_dummy:
            autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
            autoencoders.append(autoencoder)

        optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]

    # initialize feature compressor and quantizer
    feature_quantizer = ProductQuantizer(subspaces=None)
    compressor = CustomFeatureCompressor(device, feature_compression_method=feature_compression_method,
                                         quality=quality, compression_ratio=sampling_ratio,
                                         quantizer=feature_quantizer, img_size=image_size,
                                         autoencoders=autoencoders)

    print(f"Testing Dinomaly for category: {category} \n")

    if "pq" in feature_compression_method or "ae" in feature_compression_method:
        # define training dataset
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train",
                                     compressor=compressor, apply_compression=compress_images, quality=quality,
                                     img_size=image_size)
        train_dataset.load_dataset()

        # train the compressors on the training set
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
                                compressor=compressor, apply_compression=compress_images, quality=quality,
                                img_size=image_size)
    test_dataset.load_dataset()

    if feature_compression_method is not None:
        test_dataset = CompressedFeaturesDataset(feature_extractor, test_dataset, compressor, device, split="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                                      collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Length test dataset: {len(test_dataset)}")

    # define and load the model
    model, _ = create_dinomaly(backbone, device)
    model.eval()
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)

    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

    print("Evaluation performances:")
    print(f"""
        img_roc: {auroc_sp}
        pxl_roc: {auroc_px}
        f1_img: {f1_sp}
        f1_pxl: {f1_px}
        img_pr: {ap_sp}
        pxl_pr: {ap_px}
        pxl_pro: {aupro_px}
        """)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--encoder_name", type=str, default="deit_small_16", help="ViT encoder name")
    parser.add_argument("--compress_images", action="store_true", help="Compress images using JPEG or WEBP")
    parser.add_argument("--quality", type=int, default=50, help="Compression quality of images")
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+", help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Sampling ratio for random projection of features")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save/load the model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=2, help="Execution seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_dinomaly(args.dataset_path, args.category, args.encoder_name, args.save_path, device,
                       args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio,
                       args.epochs, args.seed)
    elif args.mode == "test":
        test_dinomaly(args.dataset_path, args.category, args.encoder_name, args.save_path, device,
                      args.compress_images, args.quality, args.feature_compression_method, args.sampling_ratio)


if __name__ == "__main__":
    main()
