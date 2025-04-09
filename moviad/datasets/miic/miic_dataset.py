from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL.Image import Image
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from moviad.datasets.common import IadDataset
from moviad.utilities.configurations import Split, TaskType, LabelName


class MiicDatasetClassEnum(Enum):
    """
    Enum class for MIIC dataset classes.
    """
    NORMAL = "normal"
    ANOMALY = "abnormal"


class IadDatasetConfig:
    """
    Configuration class for the IAD dataset.
    Attributes:
        task_type (TaskType): The type of task (e.g., anomaly detection).
        split (Split): The data split (e.g., train, test).
        image_shape (tuple): The shape of the images.
        mask_shape (tuple): The shape of the masks.
    """

    def __init__(self, task_type: TaskType, split: Split, image_shape: (int, int), mask_shape: (int, int),
                 preload_images: bool = False):
        self.task_type = task_type
        self.split = split
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.preload_images = preload_images


class MiicDatasetConfig(IadDatasetConfig):
    """
    Configuration class for the Miic dataset.
    Attributes:
        training_root_path (Path): Path to the training data.
        test_abnormal_image_root_path (Path): Path to the test abnormal images.
        test_normal_image_root_path (Path): Path to the test normal images.
        test_abnormal_mask_root_path (Path): Path to the test abnormal masks.
        test_abnormal_bounding_box_root_path (Path): Path to the test abnormal bounding boxes.
        task_type (TaskType): The type of task (e.g., anomaly detection).
        split (Split): The data split (e.g., train, test).
        image_shape (tuple): The shape of the images.
        mask_shape (tuple): The shape of the masks.
    """

    def __init__(self, training_root_path: Optional[Path] = None, test_abnormal_image_root_path: Optional[Path] = None,
                 test_normal_image_root_path: Optional[Path] = None,
                 test_abnormal_mask_root_path: Optional[Path] = None,
                 test_abnormal_bounding_box_root_path: Optional[Path] = None, task_type: Optional[TaskType] = None,
                 split: Optional[Split] = None, image_shape: Optional[tuple[int, int]] = (256, 256),
                 mask_shape: Optional[tuple[int, int]] = (256, 256), preload_images: bool = True):
        super().__init__(task_type, split, image_shape, mask_shape, preload_images)
        self.training_root_path = Path(training_root_path) if training_root_path else None
        self.test_abnormal_image_root_path = Path(
            test_abnormal_image_root_path) if test_abnormal_image_root_path else None
        self.test_normal_image_root_path = Path(test_normal_image_root_path) if test_normal_image_root_path else None
        self.test_abnormal_mask_root_path = Path(test_abnormal_mask_root_path) if test_abnormal_mask_root_path else None
        self.test_abnormal_bounding_box_root_path = Path(
            test_abnormal_bounding_box_root_path) if test_abnormal_bounding_box_root_path else None

    def __str__(self):
        return f"MiicDatasetConfig(training_root_path={self.training_root_path}, task={self.task}, split={self.split})"

@dataclass
class MiicDatasetEntry:
    """
    Represents a single entry in the Miic dataset.
    Example of entry path: '/root/train_root/<split>_<class_name>_<image_id>.jpg'
    """
    image_path: Path
    mask_path: str
    class_name: MiicDatasetClassEnum
    image_id: int
    split: Split
    image: Image
    mask: Image

    def __init__(self, image_path: Path, mask_path: str = None, bounding_box_path: str = None):
        image_file_name = image_path.name
        image_file_name_split = image_file_name.split('_')
        self.image_path = image_path
        self.split = Split(image_file_name_split[0])
        self.class_name = MiicDatasetClassEnum(image_file_name_split[1])
        self.image_id = int(image_file_name_split[2].split('.')[0])
        self.mask_path = mask_path
        self.bounding_box_path = bounding_box_path
        self.normal_images_entries = []
        self.abnormal_images_entries = []


class MiicDataset(IadDataset):
    def __init__(self, miic_dataset_config: MiicDatasetConfig, transform=None):
        super().__init__()
        self.config = miic_dataset_config

        self.data = []
        self.split = miic_dataset_config.split
        self.task = miic_dataset_config.task_type
        self.preload_images = self.config.preload_images
        self.category = 'semiconductor'

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.config.image_shape),
                transforms.PILToTensor(),
                transforms.Resize(
                    self.config.image_shape,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ConvertImageDtype(torch.float32),
            ])

    def set_category(self, category: str):
        self.category = category

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        image_entry = self.data[idx]
        if not self.preload_images:
            with PIL.Image.open(image_entry.image_path) as img:
                image_entry.image = img.convert("RGB")

        image = image_entry.image

        if self.split == Split.TRAIN:
            if self.transform:
                image = self.transform(image)
            return image

        label = LabelName.NORMAL.value if image_entry.class_name == MiicDatasetClassEnum.NORMAL else LabelName.ABNORMAL.value
        mask = None
        path = str(image_entry.image_path)
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                if not self.preload_images:
                    with PIL.Image.open(image_entry.mask_path) as mask_img:
                        mask = mask_img.convert("L")
                mask = self.transform(mask)
            else:
                mask = torch.zeros((1, *self.config.mask_shape), dtype=torch.float32)

        return image, label, mask, path

    def load_dataset(self):
        if self.split == Split.TRAIN:
            self.__load_training_data(self.config.training_root_path)
            return

        self.__load_test_data(self.config.test_abnormal_image_root_path,
                              self.config.test_normal_image_root_path,
                              self.config.test_abnormal_mask_root_path,
                              self.config.test_abnormal_bounding_box_root_path)

    def __load_training_data(self, normal_images_root_path: Path):
        assert normal_images_root_path.exists(), f"Normal images root path {normal_images_root_path} does not exist"
        image_file_list = list(normal_images_root_path.glob('**/*.jpg'))
        if image_file_list is None or len(image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {normal_images_root_path}")
        for image in image_file_list:
            image_entry = MiicDatasetEntry(image)
            if self.preload_images:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
            self.data.append(image_entry)
        return

    def __load_test_data(self, normal_images_root_path: Path,
                         abnormal_image_root_path: Path,
                         mask_root_path: Path,
                         bounding_box_root_path: Path):

        assert normal_images_root_path.exists(), f"Normal images root path {normal_images_root_path} does not exist"
        assert abnormal_image_root_path.exists(), f"Abnormal images root path {abnormal_image_root_path} does not exist"
        assert mask_root_path.exists(), f"Mask root path {mask_root_path} does not exist"
        assert bounding_box_root_path.exists(), f"Bounding box root path {bounding_box_root_path} does not exist"

        normal_image_file_list = list(normal_images_root_path.glob('**/*.jpg'))
        abnormal_image_file_list = list(abnormal_image_root_path.glob('**/*.jpg'))
        mask_file_list = list(mask_root_path.glob('**/*.jpg'))
        bounding_box_file_list = list(bounding_box_root_path.glob('**/*.jpg'))

        if normal_image_file_list is None or len(normal_image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {normal_images_root_path}")
        if abnormal_image_file_list is None or len(abnormal_image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {abnormal_image_root_path}")
        if mask_file_list is None or len(mask_file_list) == 0:
            raise FileNotFoundError(f"No images found in {mask_root_path}")
        for image in normal_image_file_list:
            image_entry = MiicDatasetEntry(image)
            if self.preload_images:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
            self.data.append(image_entry)

        for item in zip(abnormal_image_file_list, mask_file_list, bounding_box_file_list):
            abnormal_image_path, mask_path, bounding_box_path = item
            image_entry = MiicDatasetEntry(abnormal_image_path, mask_path, bounding_box_path)
            if self.preload_images:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
                with PIL.Image.open(mask_path) as mask:
                    image_entry.mask = mask.convert("L")
            self.data.append(image_entry)

    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> int:
        raise NotImplementedError("Dataset contamination not yet supported on this dataset.")

    def compute_contamination_ratio(self):
        """
        Compute the contamination ratio of the dataset.

        Returns:
            float: The contamination ratio.
        """
        raise NotImplementedError("Dataset contamination not yet supported on this dataset.")
