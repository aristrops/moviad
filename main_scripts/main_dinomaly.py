# This is a sample Python script.
import time

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from tqdm import tqdm

from moviad.dinomaly.dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from moviad.dinomaly.models.uad import ViTill, ViTillv2
from moviad.dinomaly.models import vit_encoder
from moviad.dinomaly.dinov1.utils import trunc_normal_
from moviad.dinomaly.models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
import torch.backends.cudnn as cudnn
import argparse
from moviad.dinomaly.utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from moviad.dinomaly.optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

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


def train(item):
    setup_seed(1)
    print_fn(item)
    num_epochs = 100
    batch_size = 16
    image_size = (448, 448)
    crop_size = 392

    #load feature extractor (encoder)
    encoder_name = 'deit_small_16'
    #encoder_name = 'dinov2reg_vit_base_14'


    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)
    encoder.to(device)

    if "ae" in args.feature_compression_method:
        with torch.no_grad():
            input_dummy = torch.randn((1, 3, 224, 224))
            features_dummy = encoder(input_dummy.to(device))

        autoencoders = nn.ModuleList()
        for layer_features in features_dummy:
            autoencoder = FeatureAutoencoder(in_channels=layer_features.shape[1], compression_ratio=0.5)
            autoencoders.append(autoencoder)

        optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in autoencoders]
    else:
        autoencoders = None

    feature_quantizer = ProductQuantizer(subspaces=None)
    compressor = CustomFeatureCompressor(device, quality=args.quality, img_size=image_size, feature_compression_method=args.feature_compression_method, compression_ratio=args.sampling_ratio, quantizer=feature_quantizer, autoencoders=autoencoders)
    
    #define training set
    train_data = MVTecDataset(TaskType.SEGMENTATION, args.data_path, item, "train", compressor, args.compress_images, args.quality, img_size=image_size)
    train_data.load_dataset()
    print(f"Length of training data: {len(train_data)}")

    if args.feature_compression_method is not None:
        if "pq" in args.feature_compression_method:
            feature_vectors = compressor.collect_feature_vectors(train_data, encoder)
            compressor.fit_quantizers(feature_vectors)
        if "ae" in args.feature_compression_method:
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
            compressor.train_autoencoders(
                train_dataloader=train_dataloader,
                feature_extractor=encoder,
                optimizers=optimizers,
                device=device,
                epochs=10,
                noise_std=0.001,
            )

        train_dataset = CompressedFeaturesDataset(encoder, train_data, compressor, device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = MVTecDataset(TaskType.SEGMENTATION, args.data_path, item, "test", compressor, args.compress_images, args.quality, img_size=image_size)
    test_data.load_dataset()
    print(f"Length of testing data: {len(test_data)}")
    if args.feature_compression_method is not None:
        if "pq" in args.feature_compression_method:
            feature_vectors = compressor.collect_feature_vectors(test_data, encoder)
            compressor.fit_quantizers(feature_vectors)
        if "ae" in args.feature_compression_method:
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
            compressor.test_reconstruction(  # Only to check overfitting, in real cases test is likely not feasible
                test_dataloader=test_dataloader,
                feature_extractor=encoder,
                device=device,
            )
        
        test_dataset = CompressedFeaturesDataset(encoder, test_data, compressor, device, split = "test")   
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)
    else:
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)


    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    total_iters = int(np.ceil(num_epochs * len(train_dataloader) / batch_size))

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    epochs = 0
    best_auroc = -float("inf")

    save_name = f"{encoder_name}_compress_images_{args.compress_images}_quality_{args.quality}.pt"
    save_dir = os.path.join(args.save_dir, item)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        start_epoch_time = time.time()

        loss_list = []

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=True
        )

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

            # Update progress bar metrics
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} finished in {epoch_time / 60:.2f} minutes")
        epochs += 1

        if (epochs + 1) % 10 == 0:
            print("Evaluating model...")
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

            if auroc_sp > best_auroc:
                best_auroc = auroc_sp
                torch.save(model.state_dict(), os.path.join(save_dir, save_name))
                print(f"New best model saved at {os.path.join(save_dir, save_name)}")

            model.train()

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_sep_dinov2br_c392_en29_bn4dp2_de8_elaelu_md2_i1_it10k_sadm2e3_wd1e4_w1hcosa_ghmp09f01w1k_b16_ev_s1')
    parser.add_argument("--quality", type=int, default=50, help="Compression quality of images")
    parser.add_argument("--compress_images", action="store_true", help="Compress images using WEBP")
    parser.add_argument("--feature_compression_method", type=str, default=None, nargs="+", help="Method for feature compression")
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Sampling ratio for random projection of features")

    args = parser.parse_args()

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    # item_list = ['leather']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = []
    for i, item in enumerate(item_list):
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = train(item)
        result_list.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
