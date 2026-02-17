import torch
import random
import io
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class CustomFeatureCompressor():
    def __init__(
        self,
        device: torch.device,
        feature_compression_method: str,
        quality: int,
        compression_ratio: int,
        quantizer = None,
        img_size = (224, 224)
    ):
        
        self.device = device
        self.quantizer = quantizer
        self.feature_compression_method = feature_compression_method
        self.quality = quality
        self.compression_ratio = compression_ratio
        self.img_size = img_size

        self.trained_quantizer = None
        self.sampled_indices_cache = {}
        self.original_shapes = None
    
    #--------FEATURE COMPRESSION--------
    
    def get_and_compress_features(self, dataset, feature_extractor, return_sizes = False):

        all_compressed_features = None
        feature_sizes = []
        compressed_features_sizes = []

        print(f"Extracting features and compressing them using {self.feature_compression_method}...")

        for image in dataset:
            #extract features
            if isinstance (image, tuple): #test set
                image_tensor = image[0].unsqueeze(0).to(self.device) #list of [1, C, H, W]
            else:
                image_tensor = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = feature_extractor(image_tensor)
            
            flat_features = torch.cat([f.flatten() for f in features])
            feature_sizes.append(flat_features.numel() * flat_features.element_size())

            features, compressed_size = self.apply_feature_compression(features)
            if compressed_size is not None:
                compressed_features_sizes.append(compressed_size)

            if all_compressed_features is None:
                all_compressed_features = [[] for _ in range(len(features))]

            for i, f in enumerate(features):
                all_compressed_features[i].append(f.squeeze(0).cpu())               
        
        if return_sizes:
            return list(zip(*all_compressed_features)), feature_sizes, compressed_features_sizes
        else:
            return [torch.stack(image_features, dim = 0) for image_features in all_compressed_features]


    def apply_feature_compression(self, features):

        total_size = None

        #random sampling
        if "random_sampling" in self.feature_compression_method:
            features = self.sample_channels(features, self.compression_ratio) #list of [1, C, H, W]

            #random sampling + webp/jpeg compression
            if "webp_for_features" in self.feature_compression_method:
                features, total_size = self.compress_features_with_codec(features, self.quality)

        #product quantization
        if "pq" in self.feature_compression_method:
            encoded = self.product_quantization(features)
            features = self.decompress_features(encoded)
        
        return features, total_size
    

    def sample_channels(self, features, compression_ratio):

        if compression_ratio == 1:
            return features
        
        reduced_features = []
        for f in features:
            B, C, H, W = f.shape
            
            key = (C, compression_ratio)
            if key not in self.sampled_indices_cache:
                k = int(C * compression_ratio) #number of channels to keep
                indices = random.sample(range(C), k)
                self.sampled_indices_cache[key] = torch.tensor(indices, device = self.device)

            sampled_idxs = self.sampled_indices_cache[key].to(self.device)
            sampled_f = torch.index_select(f, dim = 1, index = sampled_idxs)
            reduced_features.append(sampled_f)

        return reduced_features


    def compress_features_with_codec(self, features, quality):

        compressed_features = []
        total_size = 0

        for f in features:
            B, C, H, W = f.shape
            f = f.squeeze(0) #[C, H, W]

            f_min, f_max = f.min(), f.max() 
            f = (f - f_min) / (f_max - f_min + 1e-8) #normalize to [0, 255]
            f = (f * 255).clamp(0, 255).byte()

            tiles = [f[c:c+1] for c in range(C)] #[1, H, W]
            
            grid_cols = math.ceil(math.sqrt(C))
            grid_rows = math.ceil(C / grid_cols)
            
            #pad tiles to fill the grid
            pad_tiles = grid_cols * grid_rows - C
            if pad_tiles > 0:
                tiles += [torch.zeros_like(tiles[0]) for _ in range(pad_tiles)]
            
            rows = [torch.cat(tiles[r * grid_cols : (r + 1) * grid_cols], dim = 2) for r in range(grid_rows)]
            
            grid_tensor = torch.cat(rows, dim = 1).squeeze(0) #[H_total, W_total]

            img = TF.to_pil_image(grid_tensor)

            compressed_image, feature_size = self.apply_image_compression(img, quality, return_sizes = True, compress_features=True)
            total_size += feature_size

            decompressed_tensor = TF.to_tensor(compressed_image.convert("L")).squeeze(0)

            #recover tiles
            reconstructed_channels = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    y0, y1 = r * H, (r + 1) * H
                    x0, x1 = c * W, (c + 1) * W
                    reconstructed_channels.append( decompressed_tensor[y0:y1, x0:x1])

            reconstructed_tensor = torch.stack(reconstructed_channels[:C], dim = 0).unsqueeze(0).to(self.device)
            compressed_features.append(reconstructed_tensor)

        return compressed_features, total_size
    

    #--------IMAGE COMPRESSION--------

    def apply_image_compression(self, image, quality, return_sizes = False, compress_features = False):

        if not compress_features:
            image = image.resize(self.img_size)

        buffer = io.BytesIO()
        image.save(buffer, format = "WEBP", quality = quality)
        buffer.seek(0)

        if return_sizes:
            return Image.open(buffer), buffer.tell()
        else:
            return Image.open(buffer)

    
    #--------PRODUCT QUANTIZATION--------

    def collect_feature_vectors(self, dataset, feature_extractor):

        feature_vectors = [] 

        for image in dataset:
            if isinstance(image, tuple):
                features = feature_extractor(image[0].unsqueeze(0).to(self.device))
            else:
                features = feature_extractor(image.unsqueeze(0).to(self.device))
            
            if "random_sampling" in self.feature_compression_method:
                features = self.sample_channels(features, compression_ratio=self.compression_ratio) 

            #store original shapes for decoding
            self.original_shapes = [f.shape[1:] for f in features]
            
            flatten_features = torch.cat([f.flatten() for f in features])
            feature_vectors.append(flatten_features.cpu())
        
        return feature_vectors


    def fit_quantizers(self, feature_vectors):
        
        pq = self.quantizer
        X = np.stack(feature_vectors)

        print(f"Training global PQ with {X.shape[0]} vectors...")
        print(f"Shape of input matrix: {X.shape}")

        pq.fit(X)
        self.trained_quantizer = pq


    def product_quantization(self, features):

        flatten_features = torch.cat([f.flatten() for f in features])
        encoded = self.quantizer.encode(flatten_features.unsqueeze(0))
        #pq_compressed.append(torch.tensor(encoded))
        
        return encoded
        

    def decompress_features(self, compressed_features):
        decoded_features = []
        decoded = self.quantizer.decode(compressed_features).squeeze(0)
        offset = 0
        for shape_tuple in self.original_shapes:
            C, H, W = shape_tuple
            num_elements = C * H * W
            segment = decoded[offset:offset+num_elements]
            decoded_features.append(segment.reshape(C, H, W))
            offset += num_elements

        return decoded_features
        
    