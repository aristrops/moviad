import os.path
import unittest

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Subset
from torchvision.transforms import transforms, InterpolationMode

from moviad.datasets.builder import DatasetConfig, DatasetFactory, DatasetType
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadCategory, RealIadClassEnum
from moviad.entrypoints.patchcore import PatchCoreArgs
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor
from moviad.models.patchcore.kmeans_coreset_extractor import MiniBatchKMeansCoresetExtractor, KMeansCoresetExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.patchcore.product_quantizer import ProductQuantizer
from moviad.profiler.pytorch_profiler import Profiler
from moviad.trainers.batched_trainer_patchcore import BatchPatchCoreTrainer
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluator import Evaluator
from moviad.utilities.metrics import compute_product_quantization_efficiency
from tests.logger.wandb_logger import WandbLogger

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

CONFIG_PATH = '../config.json'
class PatchCoreTrainTests(unittest.TestCase):

    def setUp(self):
        self.args = PatchCoreArgs()
        self.config = DatasetConfig(CONFIG_PATH)
        dataset_factory = DatasetFactory(self.config)
        self.args.contamination_ratio = 0.25
        self.args.batch_size = 32
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)
        self.args.train_dataset = dataset_factory.build(DatasetType.Miic,split=Split.TRAIN, image_size=self.args.img_input_size)
        self.args.test_dataset = dataset_factory.build(DatasetType.Miic,split=Split.TEST, image_size=self.args.img_input_size)
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.category
        self.contamination = 0
        self.args.backbone = "mobilenet_v2"
        self.args.ad_layers = ["features.4", "features.7", "features.10"]

    def test_patchcore_coreset_clustering(self):
        k = 200
        embeddings = torch.rand([3000, 160], dtype=torch.float32)
        batch_size = 256
        batched_embeddings = torch.split(embeddings, batch_size)
        sampler = CoresetExtractor(False, self.args.device, k=k)
        coreset_kcenter = sampler.extract_coreset(embeddings.cpu())

        kmeans = MiniBatchKMeansCoresetExtractor(False, self.args.device, k=k)
        coreset_kmeans = kmeans.extract_coreset(embeddings)

        self.assertEqual(coreset_kcenter.shape[0], k + 1)
        self.assertEqual(coreset_kmeans.shape[0], k)

    def test_patchcore_quantization_efficiency(self):
        unquantized_memory_bank = torch.rand([30000, 160], dtype=torch.float32)

        pq = ProductQuantizer()
        pq.fit(unquantized_memory_bank)

        quantized_memory_bank = pq.encode(unquantized_memory_bank)

        quantization_efficiency, distortion = compute_product_quantization_efficiency(
            unquantized_memory_bank.cpu().numpy(),
            quantized_memory_bank.cpu().numpy(),
            pq)

        self.assertGreater(quantization_efficiency, 0)
        self.assertGreater(distortion, 0)

    def test_patchcore_with_quantization(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)
        training_subset = Subset(self.args.train_dataset, range(1, 1000))
        test_subset = Subset(self.args.test_dataset, range(1, 100))
        train_dataloader = torch.utils.data.DataLoader(training_subset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=True)
        trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device)

        trainer.train()
        patchcore_model.save_model("./")

    def test_patchcore_with_quantization_and_load(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)

        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=True)

        patchcore_model.load("./patchcore_model.pt", "./product_quantizer.bin")

        evaluator = Evaluator(test_dataloader, self.args.device)
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore_model)

        print("Evaluation performances:")
        print(f"""
            img_roc: {img_roc}
            pxl_roc: {pxl_roc}
            f1_img: {f1_img}
            f1_pxl: {f1_pxl}
            img_pr: {img_pr}
            pxl_pr: {pxl_pr}
            pxl_pro: {pxl_pro}
            """)

    def test_patchcore_without_quantization(self):
        profiler = Profiler()
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)
        train_dataloader = torch.utils.data.DataLoader(self.args.train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)

        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        coreset_extractor = MiniBatchKMeansCoresetExtractor(False, self.args.device, k=100)
        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=False, k=20000)
        trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device)
        profiler.start_profiling(title="PatchCore Training | K-Center greedy (k=100)")
        with profiler.profile_step():
            trainer.train()
        profiler.end_profiling()
        model_memory_size = os.path.getsize("./patchcore_model.pt")
        print(f"Model memory size: {model_memory_size}")

    def test_patchcore_streaming(self):
        profiler = Profiler()
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)
        train_dataloader = torch.utils.data.DataLoader(self.args.train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)

        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=False, k=30000)
        trainer = BatchPatchCoreTrainer(patchcore_model, train_dataloader, test_dataloader, self.args.device)
        profiler.start_profiling(title="PatchCore Training | Mini Batch K-means (k=1000)")
        with profiler.profile_step():
            trainer.train()
        profiler.end_profiling()
        model_memory_size = os.path.getsize("./patchcore_model.pt")
        print(f"Model memory size: {model_memory_size}")


if __name__ == '__main__':
    unittest.main()
