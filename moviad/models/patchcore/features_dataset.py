import torch
from torch.utils.data import Dataset

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor


class CompressedFeaturesDataset(Dataset):
    def __init__(
        self, 
        feature_extractor: CustomFeatureExtractor,
        dataset,
        compressor: CustomFeatureCompressor,
        device: torch.device,
        split: str = "train"
    ):
        
        self.feature_extractor = feature_extractor
        self.dataset = dataset
        self.compressor = compressor
        self.device = device
        self.split = split.lower()
        
        #precompute compressed features
        self.features = self.load_dataset()

    def load_dataset(self):
        return self.compressor.get_and_compress_features(self.dataset, self.feature_extractor)

    def __len__(self):
        return self.features[0].shape[0]
    
    def __getitem__(self, idx):
        sample = [layer[idx] for layer in self.features]

        if self.split == "train":
            return sample
        
        else:
            image, label, mask, path = self.dataset[idx]
            return sample, label, mask, path
    
    def collate_fn(self, batch):
        if self.split == "train":
            return [
                torch.stack([sample[i] for sample in batch])
                for i in range(len(batch[0]))
            ]
        
        else:
            features, labels, masks, paths = zip(*batch)
            stacked_features = [torch.stack([sample[i] for sample in features])
                for i in range(len(features[0]))]
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

            return stacked_features, labels, masks, paths


