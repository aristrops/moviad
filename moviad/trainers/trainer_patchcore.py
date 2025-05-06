import wandb
import torch
from sklearn.cluster import MiniBatchKMeans
import time

from tqdm import tqdm
import os

from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor
from moviad.utilities.evaluator import Evaluator


class TrainerPatchCore():

    """
    This class contains the code for training the CFA model

    Args:
        patchore_model (PatchCore): model to be trained
        train_dataloder (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        patchore_model: PatchCore,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloder: torch.utils.data.DataLoader,
        device: str,
        coreset_extractor: CoresetExtractor = None,
        logger=None
    ):
        self.patchore_model = patchore_model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloder
        self.device = device
        self.evaluator = Evaluator(self.test_dataloader, self.device)
        self.logger = logger
        self.coreset_extractor = coreset_extractor

    def train(self):

        """
        This method trains the PatchCore model and evaluate it at the end of training
        """

        embeddings = []

        with torch.no_grad():
            if self.logger is not None:
                self.logger.watch(self.patchore_model)
            print("Embedding Extraction:")
            for batch in tqdm(iter(self.train_dataloader)):
                if isinstance(batch, tuple):
                    embedding = self.patchore_model(batch[0].to(self.device))
                if isinstance(batch, list):
                    batch = [b.to(self.device) for b in batch]
                    embedding = self.patchore_model(batch)
                else:
                    embedding = self.patchore_model(batch.to(self.device))

                #print(f"Embedding Shape: {embedding.shape}")


                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)

            #print(f"Embeddings Shape: {embeddings.shape}")

            torch.cuda.empty_cache()

            # if self.patchore_model.apply_quantization:
            #     self.patchore_model.product_quantizer.fit(embeddings)
            #     embeddings = self.patchore_model.product_quantizer.encode(embeddings)


            #apply coreset reduction
            print("Coreset Extraction:")
            if self.coreset_extractor is None:
                self.coreset_extractor = CoresetExtractor(False, self.device, k=self.patchore_model.k)

            coreset = self.coreset_extractor.extract_coreset(embeddings)

            if self.patchore_model.apply_quantization:
                assert self.patchore_model.product_quantizer is not None, "Product Quantizer not initialized"

                self.patchore_model.product_quantizer.fit(coreset)
                coreset = self.patchore_model.product_quantizer.encode(coreset)

            self.patchore_model.memory_bank = coreset

            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.patchore_model)

            results = {
                    "img_roc": img_roc,
                    "pxl_roc": pxl_roc,
                    "f1_img": f1_img,
                    "f1_pxl": f1_pxl,
                    "img_pr": img_pr,
                    "pxl_pr": pxl_pr,
                    "pxl_pro": pxl_pro
                }

            # if self.logger is not None:
            #     self.logger.log({
            #         "train_loss" : 0,
            #         "val_loss" : 0,
            #         "img_roc": img_roc,
            #         "pxl_roc": pxl_roc,
            #         "f1_img": f1_img,
            #         "f1_pxl": f1_pxl,
            #         "img_pr": img_pr,
            #         "pxl_pr": pxl_pr,
            #         "pxl_pro": pxl_pro
            #     })

            if self.logger is not None:
                self.logger.log(results)

            print("End training performances:")
            print(f"""
                img_roc: {img_roc} \n
                pxl_roc: {pxl_roc} \n
                f1_img: {f1_img} \n
                f1_pxl: {f1_pxl} \n
                img_pr: {img_pr} \n
                pxl_pr: {pxl_pr} \n
                pxl_pro: {pxl_pro} \n
            """)
            
        return results

