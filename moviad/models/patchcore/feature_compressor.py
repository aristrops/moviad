import torch
import torchvision.transforms as transforms
import os
import random
import io
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import copy

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


class CustomFeatureCompressor():
    def __init__(
        self,
        device: torch.device,
        quantizer,
        compression_method: str,
        compression_ratio: int = 0.25,
        # pq_subspaces: int,
        # centroids_per_subspace: int,
        pq_method: str = None,
    ):
        
        self.device = device
        self.quantizer = quantizer
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        self.pq_method = pq_method
        # self.pq_subspaces = pq_subspaces
        # self.centroids_per_subspace = centroids_per_subspace
        self.trained_quantizer = None
        self.trained_quantizers = {}
        self.sampled_indices_cache = {}
        self.original_shapes = None
    
    def get_and_compress_features(self, dataset, feature_extractor):
        all_compressed_features = None
        compressed_features_per_image = []
        for image in dataset:
            #extract features
            if isinstance (image, tuple): #test set
                features = feature_extractor(image[0].unsqueeze(0).to(self.device)) #list of [1, C, H, W]
            else:
                features = feature_extractor(image.unsqueeze(0).to(self.device))
            
            #random sampling
            if self.compression_method == "random_sampling":
                features = self.sample_channels(features, compression_ratio=self.compression_ratio) #list of [1, C, H, W]
                if all_compressed_features is None:
                    all_compressed_features = [[] for _ in range(len(features))]
                for i, f in enumerate(features):
                    all_compressed_features[i].append(f.squeeze(0).cpu())

            #product quantization
            if self.compression_method == "pq":
                pq_features = self.product_quantization(features)
                decoded_features = self.decompress_features(pq_features)
                if all_compressed_features is None:
                    all_compressed_features = [[] for _ in range(len(decoded_features))]
                for i, f in enumerate(decoded_features):
                    all_compressed_features[i].append(f.squeeze(0).cpu())
            
        return [torch.stack(image_features, dim = 0) for image_features in all_compressed_features]

    
    #image compression
    def compress_image(self, image, compression_method = "JPEG", quality = 50, apply = False):
        if not apply:
            return image
        #print(f"Compressing images using {compression_method} with quality of {quality}...")
        compressed_io = io.BytesIO()
        image.save(compressed_io, format = compression_method, quality = quality)

        compressed_io.seek(0)

        return Image.open(compressed_io)


    #random projection of channels
    def sample_channels(self, features, compression_ratio):
        reduced_features = []
        for f in features:
            B, C, H, W = f.shape
            if compression_ratio == 1:
                reduced_features.append(f)
                continue 
            
            key = (C, compression_ratio)
            if key not in self.sampled_indices_cache:
                k = int(C * compression_ratio) #number of channels to keep
                indices = random.sample(range(C), k)
                self.sampled_indices_cache[key] = torch.tensor(indices, device = self.device)

            sampled_idxs = self.sampled_indices_cache[key].to(self.device)
            sampled_f = torch.index_select(f, dim = 1, index = sampled_idxs)
            reduced_features.append(sampled_f)
        return reduced_features
    

    def collect_feature_vectors(self, dataset, feature_extractor):
        feature_vectors = [] if self.pq_method == "global" else {}

        for image in dataset:
            if isinstance(image, tuple):
                features = feature_extractor(image[0].unsqueeze(0).to(self.device))
            else:
                features = feature_extractor(image.unsqueeze(0).to(self.device))
            
            #store original shapes for decoding
            self.original_shapes = [f.shape[1:] for f in features]
            
            if self.pq_method == "global":
                flatten_features = torch.cat([f.flatten() for f in features])
                feature_vectors.append(flatten_features.cpu())

            elif self.pq_method == "layer" or self.pq_method == "layer-channel":
                for i, feature_map in enumerate(features):

                    if self.pq_method == "layer":
                        feature_vectors.setdefault(i, []).append(feature_map.squeeze(0).flatten().numpy())
                    
                    elif self.pq_method == "layer-channel":
                        feature_map = feature_map.squeeze(0).cpu()
                        for channel in feature_map:
                            vec = channel.reshape(-1).numpy()
                            feature_vectors.setdefault(i, []).append(vec)
        
        return feature_vectors


    def fit_quantizers(self, feature_vectors):
        if self.pq_method == "global":
            pq = self.quantizer
            X = np.stack(feature_vectors)
            print(f"Training global PQ with {X.shape[0]} vectors...")
            print(f"Length of a vector: {X.shape}")
            pq.fit(X)
            self.trained_quantizer = pq

        elif self.pq_method == "layer" or self.pq_method == "layer-channel":
            for layer_idx, vectors in feature_vectors.items():
                print(f"Training PQ for layer {layer_idx} with {len(vectors)} vectors...")
                X = np.stack(vectors)
                pq = copy.deepcopy(self.quantizer)
                print(f"Shape of the matrix: {X.shape}")
                pq.fit(X)
                self.trained_quantizers[layer_idx] = pq


    def product_quantization(self, features):
        pq_compressed = None if self.pq_method == "global" else []

        if self.pq_method == "global":
            flatten_features = torch.cat([f.flatten() for f in features])
            encoded = self.quantizer.encode(flatten_features.unsqueeze(0))
            #pq_compressed.append(torch.tensor(encoded))
            pq_compressed = encoded
        
        elif self.pq_method == "layer":
            for i, feature_map in enumerate(features):
                feature_map = feature_map.squeeze(0).cpu()
                feature_map = feature_map.flatten().unsqueeze(0).numpy()
                encoded = self.trained_quantizers[i].encode(feature_map)
                pq_compressed.append(encoded)

        #     elif self.pq_method == "layer-channel":
        #         encoded_channels = []
        #         for channel in feature_map:
        #             vector = channel.reshape(1, -1).numpy()
        #             encoded = self.trained_quantizers[i].encode(vector)
        #             encoded_channels.append(torch.tensor(encoded))
        #         pq_compressed.append(torch.stack(encoded_channels))

        return pq_compressed
        

    def decompress_features(self, compressed_features):
        decoded_features = []
        if self.pq_method == "global":
            decoded = self.quantizer.decode(compressed_features).squeeze(0)
            offset = 0
            for shape_tuple in self.original_shapes:
                C, H, W = shape_tuple
                num_elements = C * H * W
                segment = decoded[offset:offset+num_elements]
                decoded_features.append(segment.reshape(C, H, W))
                offset += num_elements

        
        elif self.pq_method == "layer":
            for i, (layer_vector, shapes) in enumerate(zip(compressed_features, self.original_shapes)):
                decoded = self.trained_quantizers[i].decode(layer_vector).squeeze(0)
                C, H, W = shapes
                decoded_features.append(decoded.reshape(C, H, W))

        return decoded_features
        
    