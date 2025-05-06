import torch
import torchvision.transforms as transforms
import os
import random
import io
import math
import cv2
import tempfile
import time
import subprocess
import av
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
import copy

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


class CustomFeatureCompressor():
    def __init__(
        self,
        device: torch.device,
        image_compression_method: str,
        feature_compression_method: str,
        quality: int,
        compression_ratio: int,
        # pq_subspaces: int,
        # centroids_per_subspace: int,
        pq_method: str = None,
        quantizer = None
    ):
        
        self.device = device
        self.quantizer = quantizer
        self.image_compression_method = image_compression_method
        self.feature_compression_method = feature_compression_method
        self.quality = quality
        self.compression_ratio = compression_ratio
        self.pq_method = pq_method
        # self.pq_subspaces = pq_subspaces
        # self.centroids_per_subspace = centroids_per_subspace
        self.trained_quantizer = None
        self.trained_quantizers = {}
        self.sampled_indices_cache = {}
        self.original_shapes = None
    
    def get_and_compress_features(self, dataset, feature_extractor, return_sizes = False):
        all_compressed_features = None
        feature_sizes = []
        compressed_features_sizes = []

        for image in dataset:
            #extract features
            if isinstance (image, tuple): #test set
                features = feature_extractor(image[0].unsqueeze(0).to(self.device)) #list of [1, C, H, W]
            else:
                features = feature_extractor(image.unsqueeze(0).to(self.device))
            
            flat_features = torch.cat([f.flatten() for f in features])
            feature_size = flat_features.numel() * flat_features.element_size()
            feature_sizes.append(feature_size)
            
            if "quantize" in self.feature_compression_method:
                print("Quantizing features...")
                features = [f.to(dtype = torch.uint8) for f in features]

            #random sampling
            if "random_sampling" in self.feature_compression_method:
                features = self.sample_channels(features, compression_ratio=self.compression_ratio) #list of [1, C, H, W]

                if "jpeg_webp_for_features" in self.feature_compression_method:
                    features, compressed_features_size = self.compress_features_with_codec(features, self.image_compression_method, self.quality, mode = "grid")
                    compressed_features_sizes.append(compressed_features_size)
    
            #product quantization
            if "pq" in self.feature_compression_method:
                features = self.product_quantization(features)
                if not return_sizes: 
                    features = self.decompress_features(features)

            if all_compressed_features is None:
                all_compressed_features = [[] for _ in range(len(features))]
            for i, f in enumerate(features):
                all_compressed_features[i].append(f.squeeze(0).cpu())               
        
        if return_sizes:
            return list(zip(*all_compressed_features)), feature_sizes, compressed_features_sizes
        else:
            return [torch.stack(image_features, dim = 0) for image_features in all_compressed_features]

    
    #image compression
    def compress_image(self, image, quality, compression_method, apply = False, return_sizes = False, compress_features = False):
        if not apply:
            return image.resize((224, 224))
        #print(f"Compressing images using {compression_method} with quality of {quality}...")
        else: 
            if compress_features:
                image = image

            else: 
                image = image.resize((224, 224))

            compressed_io = io.BytesIO()

            image.save(compressed_io, format = compression_method, quality = quality)

            compressed_io.seek(0)

            if return_sizes:
                return Image.open(compressed_io), compressed_io.tell()
            else:
                return Image.open(compressed_io)


    def compress_features_with_codec(self, features, compression_method, quality, mode):
        compressed_features = []
        total_size = 0

        for f in features:
            B, C, H, W = f.shape
            f = f.squeeze(0) #[C, H, W]

            f_min, f_max = f.min(), f.max() 
            f = (f - f_min) / (f_max - f_min + 1e-8) #normalize to [0, 255]
            f = (f * 255).clamp(0, 255).byte()

            if mode == "per-channel":
                reconstructed_channels = []

                for c in range(C):
                    channel = f[c] #[H, W]

                    img = TF.to_pil_image(channel)

                    compressed_image, feature_size = self.compress_image(img, quality, compression_method, apply = True, return_sizes = True)
                    total_size += feature_size

                    decompressed_image = compressed_image.convert("L") #grayscale

                    #convert back to tensor
                    decompressed_tensor = TF.to_tensor(decompressed_image).squeeze(0) #[H, W]
                    reconstructed_channels.append(decompressed_tensor)

                #stack back to [C, H, W]
                reconstructed_tensor = torch.stack(reconstructed_channels, dim = 0).unsqueeze(0).to(self.device) #[1, C, H, W]
            
            elif mode == "grid":
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

                compressed_image, feature_size = self.compress_image(img, quality, compression_method, apply = True, return_sizes = True, compress_features=True)
                total_size += feature_size

                decompressed_image = compressed_image.convert("L")
                decompressed_tensor = TF.to_tensor(decompressed_image).squeeze(0)

                #recover tiles
                reconstructed_channels = []
                for r in range(grid_rows):
                    for c in range(grid_cols):
                        y0, y1 = r * H, (r + 1) * H
                        x0, x1 = c * W, (c + 1) * W
                        channel = decompressed_tensor[y0:y1, x0:x1]
                        reconstructed_channels.append(channel)

                reconstructed_tensor = torch.stack(reconstructed_channels[:C], dim = 0).unsqueeze(0).to(self.device)

            elif mode == "video":
                frames_np = f.cpu().numpy()
                n_frames, height, width = frames_np.shape

                if width % 2 != 0:
                    width += 1  # Make width even
                    frames_np = np.array([np.pad(frame, ((0, 0), (0, 1)), mode='constant') for frame in frames_np])  # Pad the frame
                if height % 2 != 0:
                    height += 1  # Make height even
                    frames_np = np.array([np.pad(frame, ((0, 1), (0, 0)), mode='constant') for frame in frames_np])  

                with tempfile.NamedTemporaryFile(suffix = ".mp4", delete=False) as tmpfile:
                    tmp_video_path = tmpfile.name

                output = av.open(tmp_video_path, mode = "w")
                stream = output.add_stream("libx264", rate = 24)
                stream.width = width
                stream.height = height
                stream.pix_fmt = "gray"

                for frame_np in frames_np:
                    frame_np = frame_np.astype(np.uint8)
                    video_frame = av.VideoFrame.from_ndarray(frame_np, format = "gray")
                    for packet in stream.encode(video_frame):
                        output.mux(packet)
                
                for packet in stream.encode():
                    output.mux(packet)
                output.close()

                #measure size
                feature_size = os.path.getsize(tmp_video_path)
                total_size += feature_size

                #decode the video back to tensors
                input_video = av.open(tmp_video_path)
                stream = input_video.streams.video[0]


                decoded_frames = []
                for packet in input_video.demux(stream):
                    for frame in packet.decode():
                        decoded_frames.append(frame.to_ndarray()) 

                input_video.close()
                os.remove(tmp_video_path)

                decoded_np = np.array(decoded_frames)
                decoded_tensor = torch.from_numpy(decoded_np).float()
                reconstructed_tensor = decoded_tensor.unsqueeze(0).to(self.device)   

            compressed_features.append(reconstructed_tensor)

        return compressed_features, total_size


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
            
            if "random_sampling" in self.feature_compression_method:
                features = self.sample_channels(features, compression_ratio=self.compression_ratio) 

            #store original shapes for decoding
            self.original_shapes = [f.shape[1:] for f in features]
            
            if self.pq_method == "global":
                flatten_features = torch.cat([f.flatten() for f in features])
                feature_vectors.append(flatten_features.cpu())

            elif self.pq_method == "layer" or self.pq_method == "layer-channel":
                for i, feature_map in enumerate(features):

                    if self.pq_method == "layer":
                        feature_vectors.setdefault(i, []).append(feature_map.squeeze(0).cpu().flatten())
                    
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
            print(f"Shape of input matrix: {X.shape}")
            pq.fit(X)
            self.trained_quantizer = pq

        elif self.pq_method == "layer" or self.pq_method == "layer-channel":
            for layer_idx, vectors in feature_vectors.items():
                print(f"Training PQ for layer {layer_idx} with {len(vectors)} vectors...")
                X = np.stack(vectors)
                print(f"Shape of input matrix: {X.shape}")

                pq = copy.deepcopy(self.quantizer)

                pq.fit(X)
                self.trained_quantizers[layer_idx] = pq


    def product_quantization(self, features):
        pq_compressed = None if self.pq_method == "global" else []

        if self.pq_method == "global":
            flatten_features = torch.cat([f.flatten() for f in features])
            encoded = self.quantizer.encode(flatten_features.unsqueeze(0))
            #pq_compressed.append(torch.tensor(encoded))
            pq_compressed = encoded
        
        elif self.pq_method == "layer" or self.pq_method == "layer-channel":
            for i, feature_map in enumerate(features):
                feature_map = feature_map.squeeze(0).cpu()
                if self.pq_method == "layer":
                    feature_map = feature_map.flatten().unsqueeze(0).numpy()
                    encoded = self.trained_quantizers[i].encode(feature_map)
                    pq_compressed.append(encoded)
                else:
                    encoded_channels = []
                    for channel in feature_map:
                        vec = channel.reshape(1, -1).numpy()
                        encoded = self.trained_quantizers[i].encode(vec)
                        encoded_channels.append(torch.tensor(encoded))
                    pq_compressed.append(torch.stack(encoded_channels))

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
        
        elif self.pq_method == "layer-channel":
            for i, (layer_channels, shapes) in enumerate(zip(compressed_features, self.original_shapes)):
                C, H, W = shapes
                decoded_channels = []
                for encoded_vector in layer_channels:
                    decoded = self.trained_quantizers[i].decode(encoded_vector).squeeze(0)
                    decoded_channels.append(decoded)
                
                decoded_tensor = torch.stack(decoded_channels, dim = 0)
                decoded_tensor = decoded_tensor.reshape(C, H, W)
                decoded_features.append(decoded_tensor)

        return decoded_features
        
    