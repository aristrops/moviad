from typing import Union, List
import os

from tqdm import *
import numpy as np
import torch
import torch.nn.functional as F

from moviad.models.rd4ad.rd4ad import RD4AD
from utilities.evaluator import Evaluator

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

class TrainerRD4AD:

    @staticmethod
    def loss_function(teacher_features: List[torch.Tensor], student_features: List[torch.Tensor]):

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0

        #iterate over the feature extraction layers batches
        #every feature maps shape is (B C H W)
        for i in range(len(teacher_features)):
            loss += torch.mean(
                1 - cos_loss(
                    teacher_features[i].view(teacher_features[i].shape[0],-1),
                    student_features[i].view(student_features[i].shape[0],-1)
                )
            )
        return loss

    def __init__(
        self,
        r4ad_model: RD4AD,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloder: torch.utils.data.DataLoader,
        device: str,
        logger=None
    ):
        self.model = r4ad_model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloder
        self.device = device
        self.evaluator = Evaluator(self.test_dataloader, self.device)
        self.logger = logger

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        self.model.train()

        learning_rate = 0.005

        optimizer = torch.optim.Adam(
            list(self.model.decoder.parameters())+list(self.model.bn.parameters()),
            lr=learning_rate,
            betas=(0.5,0.999)
        )

        best_img_roc = 0
        best_pxl_roc = 0
        best_img_f1 = 0
        best_pxl_f1 = 0
        best_img_pr = 0
        best_pxl_pr = 0
        best_pxl_pro = 0

        for epoch in trange(epochs):

            print(f"EPOCH: {epoch}")

            self.model.train()

            #train the model
            batch_loss = 0
            for batch in tqdm(self.train_dataloader):

                batch = batch.to(self.device)
                teacher_features, bn_features, student_features = self.model(batch)

                loss = TrainerRD4AD.loss_function(teacher_features, student_features)

                batch_loss += loss.item()
                if self.logger: 
                    self.logger.log({"loss": loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_batch_loss = batch_loss / len(self.train_dataloader)
            if self.logger:
                self.logger.log({"avg_batch_loss": avg_batch_loss})
            print(f"Avg loss on epoch {epoch}: {avg_batch_loss}")

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.model)

                if self.logger:
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

        return best_results
