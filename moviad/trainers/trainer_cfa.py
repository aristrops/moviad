import os

import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm

from moviad.models.cfa.cfa import CFA
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluator import Evaluator


class TrainerResult:
    img_roc: float
    pxl_roc: float
    f1_img: float
    f1_pxl: float
    img_pr: float
    pxl_pr: float
    pxl_pro: float

    def __init__(self, img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro):
        self.img_roc = img_roc
        self.pxl_roc = pxl_roc
        self.f1_img = f1_img
        self.f1_pxl = f1_pxl
        self.img_pr = img_pr
        self.pxl_pr = pxl_pr
        self.pxl_pro = pxl_pro


class TrainerCFA():
    """
    This class contains the code for training the CFA model

    Args:
        cfa_model (CFA): model to be trained
        backbone (str) : feature extractor backbone
        feature_extractor (CustomFeatureExtractor): feature extractor to be used
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        category (str): mvtec category
        device (str): device to be used for the training
    """

    def __init__(
            self,
            cfa_model: CFA,
            backbone: str,
            feature_extractor: CustomFeatureExtractor,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloder: torch.utils.data.DataLoader,
            category: str,
            device: str,
            logger=None
    ):
        self.cfa_model = cfa_model
        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloder
        self.device = device
        self.evaluator = Evaluator(self.test_dataloader, self.device)
        self.category = category
        self.logger = logger

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):
        """
        Train the model by first extracting the features from the batches, transform them 
        with the patch descriptor and then apply the CFA loss

        Args:
            evaluation_epoch_interval: optional, number of epochs between evaluations
            epochs (int) : number of epochs for the training
            save_model (bool) : true if the model must be saved at the end of training 
            visual_test (bool) : true if we want to produce test images with heatmaps
        """

        params = [{'params': self.cfa_model.parameters()}, ]
        learning_rate = 1e-3
        weight_decay = 5e-4
        optimizer = AdamW(params=params,
                          lr=learning_rate,
                          weight_decay=weight_decay,
                          amsgrad=True)

        best_img_roc = img_roc = 0
        best_pxl_roc = pxl_roc = 0
        best_img_f1 = f1_img = 0
        best_pxl_f1 = f1_pxl = 0
        best_img_pr = img_pr = 0
        best_pxl_pr = pxl_pr = 0
        best_pxl_pro = pxl_pro = 0

        if self.logger is not None:
            self.logger.config.update(
                {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "category": self.category,
                    "optimizer": "AdamW"
                },
                allow_val_change=True
            )
            self.logger.watch(self.cfa_model, log='all', log_freq=10)
        self.cfa_model.train()

        for epoch in range(epochs):

            print(f"EPOCH: {epoch}")

            self.cfa_model.train()
            batch_loss = 0
            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()

                """
                p = self.feature_extractor(batch.to(self.device))
                if isinstance(p, dict):
                    p = list(p.values())
                    
                loss, _ = self.cfa_model(p)
                """
                loss = self.cfa_model(batch.to(self.device))
                batch_loss += loss.item()
                if self.logger is not None:
                    self.logger.log({"loss": loss.item()})
                loss.backward()
                optimizer.step()

            avg_batch_loss = batch_loss / len(self.train_dataloader)
            if self.logger is not None:
                self.logger.log({"avg_batch_loss": avg_batch_loss})

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.cfa_model)

                if self.logger is not None:
                    self.logger.log({
                        "img_roc": img_roc,
                        "pxl_roc": pxl_roc,
                        "f1_img": f1_img,
                        "f1_pxl": f1_pxl,
                        "img_pr": img_pr,
                        "pxl_pr": pxl_pr,
                        "pxl_pro": pxl_pro
                    })

                best_img_roc = img_roc if img_roc > best_img_roc else best_img_roc
                best_pxl_roc = pxl_roc if pxl_roc > best_pxl_roc else best_pxl_roc
                best_img_f1 = f1_img if f1_img > best_img_f1 else best_img_f1
                best_pxl_f1 = f1_pxl if f1_pxl > best_pxl_f1 else best_pxl_f1
                best_img_pr = img_pr if img_pr > best_img_pr else best_img_pr
                best_pxl_pr = pxl_pr if pxl_pr > best_pxl_pr else best_pxl_pr
                best_pxl_pro = pxl_pro if pxl_pro > best_pxl_pro else best_pxl_pro

                print("Mid training performances:")
                print(f"""
                    img_roc: {img_roc} \n
                    pxl_roc: {pxl_roc} \n
                    f1_img: {f1_img} \n
                    f1_pxl: {f1_pxl} \n
                    img_pr: {img_pr} \n
                    pxl_pr: {pxl_pr} \n
                    pxl_pro: {pxl_pro} \n
                """)

        print("Best training performances:")
        print(f"""
                img_roc: {best_img_roc} \n
                pxl_roc: {best_pxl_roc} \n
                f1_img: {best_img_f1} \n
                f1_pxl: {best_pxl_f1} \n
                img_pr: {best_img_pr} \n
                pxl_pr: {best_pxl_pr} \n
                pxl_pro: {best_pxl_pro} \n
        """)

        best_results = TrainerResult(
            img_roc=best_img_roc,
            pxl_roc=best_pxl_roc,
            f1_img=best_img_f1,
            f1_pxl=best_pxl_f1,
            img_pr=best_img_pr,
            pxl_pr=best_pxl_pr,
            pxl_pro=best_pxl_pro
        )

        results = TrainerResult(
            img_roc=img_roc,
            pxl_roc=pxl_roc,
            f1_img=f1_img,
            f1_pxl=f1_pxl,
            img_pr=img_pr,
            pxl_pr=pxl_pr,
            pxl_pro=pxl_pro
        )

        return results, best_results
