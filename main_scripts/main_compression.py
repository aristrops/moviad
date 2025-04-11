import torch
import torchvision.transforms as transforms
import os
import random
import io
import numpy as np
from PIL import Image
import argparse
import sys

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.configurations import TaskType
from moviad.models.patchcore.product_quantizer import ProductQuantizer


def compress_features(dataset_path, categories, device, backbone, layer_idxs, compression_method, 
                      quality, feature_dtype, pq_method, pq_subspaces, centroids_per_subspace, random_sampling, sampling_ratio):
    
    def preprocess_image(image):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform(image).unsqueeze(0).to(device)

    def get_feature_vectors(features, method):
        vectors = {}
        if method == "layer":
            for idx, fmap in zip(layer_idxs, features):
                vectors.setdefault(idx, []).append(fmap.squeeze(0).cpu().flatten())
        elif method == "layer-channel":
            for idx, fmap in zip(layer_idxs, features):
                fmap = fmap.squeeze(0).cpu()
                for channel in fmap:
                    vectors.setdefault(idx, []).append(channel.reshape(-1).numpy())
        return vectors

    def sample_channels(features, ratio = 0.25):
        reduced_features = []
        for f in features:
            B, C, H, W = f.shape
            k = int(C * ratio) #number of channels to keep
            sampled_idxs = torch.tensor(random.sample(range(C), k), device = f.device)
            sampled_f = torch.index_select(f, dim = 1, index = sampled_idxs)
            reduced_features.append(sampled_f.flatten())
        return torch.cat(reduced_features)

    
    print(f"Compressing images using {compression_method} with quality of {quality}")
    print(f"Casting features to {feature_dtype}")

    for category in categories:
        print(f"Processing {category} category")

        #load dataset
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, dataset_path, category, "train")
        train_dataset.load_dataset()
        dataset_images = train_dataset.samples.sample(len(train_dataset))
        print(f"Number of images to compress: {len(train_dataset)}")
    
        feature_extractor = CustomFeatureExtractor(backbone, layer_idxs, device = device)

        features_list, feature_vectors = [], {idx: [] for idx in layer_idxs}

        sizes = {"original": [], "compressed": [], "features": [], "quantized_features": []}

        all_features = []

        #collect features
        print("Extracting features...")
        for _, sample in dataset_images.iterrows():
            image_path = sample["image_path"]
            original_image = Image.open(image_path).convert("RGB")

            #apply transformations
            image_tensor = preprocess_image(original_image)

            #feature extraction
            features = feature_extractor(image_tensor) #list of [1, C, H, W]
            if random_sampling:
                flat_features = sample_channels(features, sampling_ratio).to(dtype = feature_dtype)
            else:
                flat_features = torch.cat([f.flatten() for f in features]).to(dtype=feature_dtype)
            features_list.append(flat_features.cpu())
            all_features.append((original_image, features, flat_features))

            if pq_method in ["layer", "layer-channel"]:
                vectors = get_feature_vectors(features, pq_method)
                for idx, vecs in vectors.items():
                    feature_vectors[idx].extend(vecs)

        #train product quantizers
        if pq_method == "global":
            pq = ProductQuantizer(subspaces = pq_subspaces, centroids_per_subspace= centroids_per_subspace)
            feature_matrix = torch.stack(features_list)  
            print(f"Training global PQ with {feature_matrix.shape[0]} vectors...")
            pq.fit(feature_matrix)  
        elif pq_method == "layer" or pq_method == "layer-channel":
            trained_quantizers = {}
            for idx, vectors in feature_vectors.items():
                print(f"Training PQ for layer {idx} with {len(vectors)} vectors...")
                X = np.stack(vectors)
                layer_pq = ProductQuantizer(subspaces = pq_subspaces, centroids_per_subspace = centroids_per_subspace)
                layer_pq.fit(X)
                trained_quantizers[idx] = layer_pq

        #encode and measure sizes
        print("Encoding features and measuring sizes...")
        for img, features, flat_features in all_features:
            #compute the original size
            original_size = os.path.getsize(image_path)
            sizes["original"].append(original_size)

            #compress image
            compressed_image_io = io.BytesIO()
            img.save(compressed_image_io, format=compression_method, quality = quality) 
            sizes["compressed"].append(compressed_image_io.tell())

            #compute feature size
            feature_size = flat_features.numel() * flat_features.element_size()
            sizes["features"].append(feature_size)

            #encode and measure size of compressed features
            pq_size = 0
            if pq_method == "global":
                compressed_feature = pq.encode(flat_features.unsqueeze(0)) 
                pq_size = compressed_feature.numel() * compressed_feature.element_size()
            elif pq_method == "layer" or pq_method == "layer-channel":
                for layer_idx, feature_map in zip(layer_idxs, features):
                    feature_map = feature_map.squeeze(0).cpu()
                    if pq_method == "layer":
                        encoded = trained_quantizers[layer_idx].encode(feature_map.flatten().unsqueeze(0))
                    else:
                        for channel in feature_map:
                            encoded = trained_quantizers[layer_idx].encode(channel.reshape(1, -1).numpy())
                    pq_size += encoded.numel() * encoded.element_size()
            sizes["quantized_features"].append(pq_size)
        
        def print_comparisons(label, key1, key2 = None):
            base = sum(sizes[key1])
            comparison = sum(sizes[key2]) if key2 else base
            print(f"{label}: {1-(comparison/base): .2%}")
        
        #print results
        #print(f"Average original size: {int(avg_original_size)} bytes")
        #print(f"Average compressed size: {int(avg_compressed_size)} bytes")
        print_comparisons(f"Compression ratio of images", "original", "compressed")

        print(f"Average feature size: {int(np.mean(sizes["features"]))} bytes")
        print_comparisons(f"Compression ratio of features/images", "original", "features")

        if pq_method:
            print(f"Average PQ feature size: {int(np.mean(sizes["quantized_features"]))} bytes")
            print_comparisons(f"Compression ratio using PQ features", "features", "quantized_features")
        
        print(f"Size of the product quantizers: {sys.getsizeof(trained_quantizers)} bytes")

    return



def main():

    categories = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", 
                  "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type = str, help = "Path of the directory where the dataset is stored")
    parser.add_argument("--categories", type = str, nargs = "+", default = categories, help = "Dataset categories to perform compression on")
    parser.add_argument("--device", type = str, help = "Where to run the script")
    parser.add_argument("--backbone", type = str, help = "CNN model to use for feature extraction")
    parser.add_argument("--layer_idxs", type = str, nargs = "+", help = "List of layers to use for extraction")
    parser.add_argument("--compression_method", type = str, default = "JPEG", help = "Compression method to use on images, e.g. JPEG, PNG, ...")
    parser.add_argument("--quality", type = int, default = 50, help = "Amount of compression to be applied to the image when compressing with JPEG")
    parser.add_argument("--feature_dtype", type = str, default = "float32", help = "Data type of the features")
    parser.add_argument("--pq_method", type = str, default = None, help = "Choose whether to perform product quantization globally, layer-wise or channel- and layer-wise. By default it doesn't apply compression")
    parser.add_argument("--pq_subspaces", type = int, help = "Number of subspaces to use in product quantization")
    parser.add_argument("--centroids_per_subspace", type = int, default = 256, help = "Number of centroids per subspace to be used in PQ") 
    parser.add_argument("--random_sampling", action = "store_true", help = "Applies random sampling of the features to reduce the dimensionality")
    parser.add_argument("--sampling_ratio", type = float, default = 0.25, help = "Percentage of features that have to be kept")
    parser.add_argument("--seed", type = int, default = 1, help = "Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64}

    pq_methods = [None, "global", "layer", "layer-channel"]

    if args.feature_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported feature_dtype: {args.feature_dtype}. Choose from {list(dtype_mapping.keys())}")

    if args.pq_method not in pq_methods:
        raise ValueError(f"Unsupported PQ method: {args.pq_method}. Choose from {pq_methods}")

    args.feature_dtype = dtype_mapping[args.feature_dtype]

    compress_features(args.dataset_path, args.categories, device, args.backbone, args.layer_idxs, args.compression_method, 
                      args.quality, args.feature_dtype, args.pq_method, args.pq_subspaces, args.centroids_per_subspace, args.random_sampling, args.sampling_ratio)


if __name__ == "__main__":
    main()






