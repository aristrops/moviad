import json
import os
from enum import EnumType
from moviad.datasets.iad_dataset import IadDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.utilities.configurations import TaskType, Split


class DatasetConfig:
    def __init__(self, config_file, image_size=(256, 256)):
        self.config = self.load_config(config_file)
        self.realiad_root_path = self.convert_path(self.config['datasets']['realiad']['root_path'])
        self.realiad_json_root_path = self.convert_path(self.config['datasets']['realiad']['json_root_path'])
        self.visa_root_path = self.convert_path(self.config['datasets']['visa']['root_path'])
        self.visa_csv_path = self.convert_path(self.config['datasets']['visa']['csv_path'])
        self.mvtec_root_path = self.convert_path(self.config['datasets']['mvtec']['root_path'])
        self.image_size = image_size

    def load_config(self, config_file):
        assert os.path.exists(config_file), f"Config file {config_file} does not exist"
        ext = os.path.splitext(config_file)[1].lower()
        if ext == '.yaml' or ext == '.yml':
            return self.load_yaml_config(config_file)
        elif ext == '.json':
            return self.load_json_config(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    def load_json_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def convert_path(self, path):
        return os.path.normpath(path)

class DatasetType(EnumType):
    MVTec = "mvtec"
    RealIad = "realiad"
    Visa = "visa"

class DatasetFactory:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.image_size = (256, 256)

    def build(self, dataset_type: DatasetType, split: Split, class_name: str, image_size=(256, 256)) -> IadDataset:
        """
        Build a dataset based on the provided type and parameters.
        :param dataset_type: The type of dataset to build.
        :param split: The split of the dataset (train/test).
        :param class_name: The class name for the dataset.
        :param image_size: The size of the images in the dataset. Defaults to (256, 256).
        :return: An instance of the dataset.
        """
        if dataset_type == DatasetType.MVTec:
            return MVTecDataset(
                TaskType.SEGMENTATION,
                self.config.mvtec_root_path,
                class_name,
                split,
                img_size=image_size,
                gt_mask_size=image_size
            )
        elif dataset_type == DatasetType.RealIad:
            return RealIadDataset(
                class_name,
                self.config.realiad_root_path,
                self.config.realiad_json_root_path,
                task=TaskType.SEGMENTATION,
                split=split,
                image_size=image_size,
                gt_mask_size=image_size
            )
        elif dataset_type == DatasetType.Visa:
            return VisaDataset(
                self.config.visa_root_path,
                self.config.visa_csv_path,
                split=split,
                class_name=class_name,
                image_size=image_size,
                gt_mask_size=image_size
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")