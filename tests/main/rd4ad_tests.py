import unittest

import torch

from moviad.datasets.builder import DatasetConfig, DatasetFactory, DatasetType
from moviad.entrypoints.rd4ad import RD4ADArgs, train_rd4ad
from moviad.utilities.configurations import Split


class Rd4adTests(unittest.TestCase):
    def setUp(self):
        self.args = RD4ADArgs()
        self.config = DatasetConfig('../config.json')
        dataset_factory = DatasetFactory(self.config)
        self.args.contamination_ratio = 0.25
        self.args.batch_size = 32
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)
        self.args.dataset_config = self.config
        self.args.dataset_type = DatasetType.MVTec
        self.args.category = 'bottle'
        self.contamination = 0
        self.args.backbone = "mobilenet_v2"
        self.args.ad_layers = ["features.4", "features.7", "features.10"]


    def test_rd4ad(self):
        train_rd4ad(self.args)