import torch
import os
import random
import cv2
import io
import numpy as np
from PIL import Image
import argparse


from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.features_dataset import CustomFeatureCompressor
from moviad.utilities.configurations import TaskType, LabelName
from moviad.models.patchcore.product_quantizer import ProductQuantizer


def size_computing(dataset_path, categories, device, backbone, layer_idxs, image_compression_method, quality, 
                   feature_compression_method, sampling_ratio, pq_method, pq_subspaces, centroids_per_subspace):
    
    for category in categories:

        sizes = {"original": [], "compressed": [], "features": [], "compressed_features": []}

        print(f"Processing {category} category")

        quantizer = ProductQuantizer(subspaces=976)
        
        compressor = CustomFeatureCompressor(device, image_compression_method=image_compression_method, feature_compression_method = feature_compression_method, 
                                quality = quality, compression_ratio = sampling_ratio, pq_method = pq_method, quantizer=quantizer)
        feature_extractor = CustomFeatureExtractor(backbone, layer_idxs, device = device)
        
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train", compressor = compressor, apply_compression=False)
        train_dataset.load_dataset()
        dataset_images = train_dataset.samples.sample(len(train_dataset))
        print(f"Number of images in the dataset: {len(train_dataset)}")

        test_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "test", compressor = compressor, apply_compression=False)
        test_dataset.load_dataset()

        for _, sample in dataset_images.iterrows():
            image_path = sample["image_path"]
            original_image = Image.open(image_path).convert("RGB")
            resized_image = original_image.resize((224, 224)).convert("RGB")
            with io.BytesIO() as byte_io:
                resized_image.save(byte_io, format = "PNG", quality = "keep")
                resized_image_size = byte_io.tell()
            sizes["original"].append(resized_image_size)

            compressed_image, compressed_size = compressor.compress_image(original_image, quality = quality, compression_method = image_compression_method, apply = True, return_sizes = True)
            sizes["compressed"].append(compressed_size)
        
        if "pq" in feature_compression_method:
            feature_vectors = compressor.collect_feature_vectors(train_dataset, feature_extractor)
            compressor.fit_quantizers(feature_vectors)
        print(f"Compressing features with {feature_compression_method}...")
        compressed_features, original_feature_sizes, compressed_feature_sizes = compressor.get_and_compress_features(train_dataset, feature_extractor, return_sizes = True)
        sizes["features"] = original_feature_sizes
        if "jpeg_webp_for_features" in feature_compression_method:
            sizes["compressed_features"] = compressed_feature_sizes
        else:
            for feature_list in compressed_features:
                flat_features = torch.cat([f.flatten() for f in feature_list])
                compressed_feature_size = flat_features.numel() * flat_features.element_size()
                sizes["compressed_features"].append(compressed_feature_size)


        def print_comparisons(key1, key2 = None):
            base = sum(sizes[key1])
            comparison = sum(sizes[key2]) if key2 else base
            if comparison < base:
                print(f"Space saved with {key2} over {key1}: {1-(comparison/base): .2%}")
            else:
                print(f"Space lost with {key2} over {key1}: {(comparison/base)-1: .2%}")
        
        anomalous_test_samples = test_dataset.samples[(test_dataset.samples.split == "test") & (test_dataset.samples.label_index == LabelName.ABNORMAL)]
        bounding_box_areas = []

        for idx, row in anomalous_test_samples.iterrows():
            mask_path = row["mask_path"]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

            for stat in stats[1:]: #skip the background
                x, y, w, h, area = stat 
                bounding_box_areas.append(area)  

        avg_original_size = np.mean(sizes["original"])
        avg_compressed_size = np.mean(sizes["compressed"])
        avg_feature_size = np.mean(sizes["features"])
        avg_compressed_feature_size = np.mean(sizes["compressed_features"])
        avg_bounding_box_area = np.mean(bounding_box_areas)

        print(f"Average original size: {int(avg_original_size/1000)} kilobytes")
        print(f"Average compressed size: {int(avg_compressed_size)} bytes")
        print_comparisons("original", "compressed")

        print(f"Average feature size: {int(avg_feature_size/1000)} kilobytes")
        print_comparisons("original", "features")

        print(f"Average compressed feature size:  {int(avg_compressed_feature_size/1000)} kilobytes")
        print_comparisons("features", "compressed_features")
        print_comparisons("original", "compressed_features")
        print_comparisons("compressed", "compressed_features")

        print(f"Average area of the bounding box of the mask: {int(avg_bounding_box_area)} pixels")


def main():

    categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", 
                  "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type = str, help = "Path of the directory where the dataset is stored")
    parser.add_argument("--categories", type = str, nargs = "+", default = categories, help = "Dataset categories to perform compression on")
    parser.add_argument("--device", type = str, help = "Where to run the script")
    parser.add_argument("--backbone", type = str, help = "CNN model to use for feature extraction")
    parser.add_argument("--layer_idxs", type = str, nargs = "+", help = "List of layers to use for extraction")
    parser.add_argument("--image_compression_method", type = str, default = "JPEG", help = "Compression method to use on images, e.g. JPEG, PNG, ...")
    parser.add_argument("--quality", type = int, default = 50, help = "Amount of compression to be applied to the image when compressing with JPEG")
    parser.add_argument("--feature_compression_method", type = str, default = None, nargs = "+", help = "Compression method to use on features, e.g. random sampling, ...")
    parser.add_argument("--sampling_ratio", type = float, default = 0.25, help = "Percentage of features that have to be kept")
    parser.add_argument("--pq_method", type = str, default = None, help = "Choose whether to perform product quantization globally, layer-wise or channel- and layer-wise. By default it doesn't apply compression")
    parser.add_argument("--pq_subspaces", type = int, default = None, help = "Number of subspaces to use in product quantization")
    parser.add_argument("--centroids_per_subspace", type = int, default = None, help = "Number of centroids per subspace to be used in PQ") 
    parser.add_argument("--seed", type = int, default = 1, help = "Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    pq_methods = [None, "global", "layer", "layer-channel"]

    if args.pq_method not in pq_methods:
        raise ValueError(f"Unsupported PQ method: {args.pq_method}. Choose from {pq_methods}")

    size_computing(args.dataset_path, args.categories, device, args.backbone, args.layer_idxs, args.image_compression_method, 
                      args.quality, args.feature_compression_method, args.sampling_ratio, args.pq_method, args.pq_subspaces, args.centroids_per_subspace)
    

if __name__ == "__main__":
    main()