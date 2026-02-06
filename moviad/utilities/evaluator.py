from __future__ import annotations
import os
import psutil
from typing import Union, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch

from ..utilities.metrics import *


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    def __init__(self, test_dataloader, device):
        """
        Args:
            test_dataloader (Dataloader): test dataloader, the images should already be normalized
            device (torch.device): device where to run the model
        """
        self.test_dataloader = test_dataloader
        self.device = device

    def evaluate(self, model,  logger = None, output_path = False):
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            output_path (str): path where to store the output masks
        """

        model.eval()

        # Initialize results.
        gt_masks_list, true_img_scores = (list(), list())
        pred_masks, pred_img_scores = (list(), list())

        for images, labels, masks, path in tqdm(self.test_dataloader, desc="Eval"):
            # get anomaly map and score
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model(images)

            if anomaly_maps.shape[2:] != masks.shape[2:]:
                raise Exception(
                    "The output anomaly maps should have the same resolution as the target masks."
                    + f"Expected shape: {masks.shape[2:]}, got: {anomaly_maps.shape[2:]}"
                )
            
            # add true masks and img anomaly scores
            gt_masks_list.extend(masks.cpu().numpy().astype(int))
            true_img_scores.extend(labels.cpu().numpy())

            # add predicted masks and img anomaly scores (check for numpy arrays or tensors)
            if isinstance(anomaly_maps, torch.Tensor):
                pred_masks.extend(anomaly_maps.cpu().numpy())
                pred_img_scores.extend(anomaly_scores.cpu().numpy())
            else:
                pred_masks.extend(anomaly_maps)
                pred_img_scores.extend(anomaly_scores)

        gt_masks_list = np.asarray(gt_masks_list)
        true_img_scores = np.asarray(true_img_scores)
        pred_masks = np.asarray(pred_masks)
        pred_img_scores = np.asarray(pred_img_scores)

        pred_masks = min_max_norm(pred_masks)

        """Image-level AUROC"""
        fpr, tpr, img_roc_auc = cal_img_roc(pred_img_scores, true_img_scores)

        """Pixel-level AUROC"""
        fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_masks_list, pred_masks)

        """F1 Score Image-level"""
        f1_img = cal_f1_img(pred_img_scores, true_img_scores)

        """F1 Score Pixel-level"""
        f1_pxl = cal_f1_pxl(pred_masks, gt_masks_list)

        """Image-level PR-AUC"""
        pr_auc_img = cal_pr_auc_img(pred_img_scores, true_img_scores)

        """Pixel-level PR-AUC"""
        pr_auc_pxl = cal_pr_auc_pxl(pred_masks, gt_masks_list)

        """Pixel-level AU-PRO"""
        au_pro_pxl = cal_pro_auc_pxl(np.squeeze(pred_masks, axis=1), gt_masks_list)

        # TODO: Implement Add False-alarm rate

        if logger is not None:
            logger.log({
                "img_roc_auc": img_roc_auc,
                "per_pixel_rocauc": per_pixel_rocauc,
                "f1_img": f1_img,
                "f1_pxl": f1_pxl,
                "pr_auc_img": pr_auc_img,
                "pr_auc_pxl": pr_auc_pxl,
                "au_pro_pxl": au_pro_pxl
            })

        return (
            img_roc_auc,
            per_pixel_rocauc,
            f1_img,
            f1_pxl,
            pr_auc_img,
            pr_auc_pxl,
            au_pro_pxl,
        )

           
    def evaluate_single_images(self, model):

        model.eval()

        # compute the threshold as equal precision and recall on the test dataset
        pred_anom_score_lst, true_anom_score_lst = [], []
        pred_anom_map_lst, gt_anom_mask_lst = [], []
        allpaths = []
        for images, labels, masks, paths in tqdm(self.test_dataloader):
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model(images.to(self.device))

            if isinstance(anomaly_maps, torch.Tensor):
                anomaly_maps = anomaly_maps.cpu().numpy()
                anomaly_scores = anomaly_scores.cpu().numpy()

            gt_masks_list = masks.cpu().numpy().astype(int)
            true_img_scores = labels.cpu().numpy()

            pred_anom_score_lst.extend(anomaly_scores)
            true_anom_score_lst.extend(true_img_scores)
            pred_anom_map_lst.extend(anomaly_maps)
            gt_anom_mask_lst.extend(gt_masks_list)
            allpaths.extend(paths)

        pred_anom_score_lst = np.asarray(pred_anom_score_lst)
        true_anom_score_lst = np.asarray(true_anom_score_lst)
        pred_anom_map_lst = np.asarray(pred_anom_map_lst)
        gt_anom_mask_lst = np.asarray(gt_anom_mask_lst)

        pred_anom_map_lst = min_max_norm(pred_anom_map_lst)

        # the threshold is the value that minimizes the difference between precision and recall
        precision, recall, thresholds = precision_recall_curve(
            gt_anom_mask_lst.flatten(), pred_anom_map_lst.flatten()
        )
        threshold = thresholds[np.argmin(np.abs(precision - recall))]

        pred_mask_lst = (pred_anom_map_lst > threshold).astype(int)

        print(
            len(pred_anom_score_lst),
            len(true_anom_score_lst),
            len(pred_mask_lst),
            len(gt_anom_mask_lst),
            len(allpaths),
        )

        metrics = []
        for pred_anom_score, true_anom_score, pred_anom_mask, gt_anom_mask, path in zip(
            pred_anom_score_lst,
            true_anom_score_lst,
            pred_mask_lst,
            gt_anom_mask_lst,
            allpaths,
        ):
            gt_anom_mask = gt_anom_mask.flatten()
            pred_anom_mask = pred_anom_mask.flatten()

            precision = precision_score(gt_anom_mask, pred_anom_mask, zero_division=0)
            recall = recall_score(gt_anom_mask, pred_anom_mask, zero_division=0)
            f1 = f1_score(gt_anom_mask, pred_anom_mask, zero_division=0)

            false_alarm_rate = np.sum(
                (gt_anom_mask == 0) & (pred_anom_mask == 1)
            ) / np.sum(gt_anom_mask == 0)

            metrics.append(
                {
                    "pred_anom_score": pred_anom_score,
                    "true_anom_score": true_anom_score,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_alarm_rate": false_alarm_rate,
                    "path": path,
                }
            )
        metrics = pd.DataFrame(metrics)

        return metrics, threshold, gt_anom_mask_lst, pred_anom_map_lst


    @staticmethod
    def get_threshold(gt: np.ndarray, score: np.ndarray) -> float:
        """
        Calculate the segmentation threshold

        Args:
            gt (np.array)    : ground truth masks
            score (np.array) : predicted masks

        Returns:
            threshold (float) : segmentation threshold
        """

        gt_mask = np.asarray(gt)
        precision, recall, thresholds = precision_recall_curve(
            gt_mask.flatten(), score.flatten()
        )
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        # consider the threshold with the highest f1 score
        threshold = thresholds[np.argmax(f1)]

        return threshold

    def measure_peak_inference_memory(self, model):
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for batch in self.test_dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(self.device)

                _ = model(inputs)
                break  

        peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return peak_mem_mb
    
    
    def measure_peak_inference_memory_cpu(self, model):
        """
        Measure peak CPU memory usage (in MB) during a single forward pass.

        Args:
            model: the PatchCore or any PyTorch model

        Returns:
            peak_cpu_mem_mb: Peak CPU memory in MB
        """
        model.eval()
        process = psutil.Process(os.getpid())
        
        # Record initial memory
        mem_before = process.memory_info().rss / (1024 ** 2)  # in MB
        peak_mem = mem_before

        with torch.no_grad():
            for batch in self.test_dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                # Move to CPU explicitly
                inputs = inputs.cpu()

                _ = model(inputs)  # forward pass

                mem_now = process.memory_info().rss / (1024 ** 2)
                peak_mem = max(peak_mem, mem_now)
                
                break  # measure only single batch

        peak_cpu_mem_mb = peak_mem - mem_before
        return peak_cpu_mem_mb
    
def append_results(
    output_path: Union[str, os.PathLike],
    category: str,
    seed: Optional[int],
    img_roc_auc: float,
    per_pixel_rocauc: float,
    f1_img: float,
    f1_pxl: float,
    pr_auc_img: float,
    pr_auc_pxl: float,
    au_pro_pxl: float,
    ad_model: str,
    feature_layers: str,
    backbone: str,
    weights: Optional[str],
    bootstrap_layer: Optional[int],
    epochs: Optional[int],
    input_img_size: Optional[tuple[int, int]],
    output_img_size: Optional[tuple[int, int]],
):
    """
    Save the results of the evaluation in a file
    """
    df = pd.DataFrame(
        {
            "category": [category],
            "seed": [seed],
            "img_roc_auc": [img_roc_auc],
            "per_pixel_rocauc": [per_pixel_rocauc],
            "f1_img": [f1_img],
            "f1_pxl": [f1_pxl],
            "pr_auc_img": [pr_auc_img],
            "pr_auc_pxl": [pr_auc_pxl],
            "au_pro_pxl": [au_pro_pxl],
            "ad_model": [ad_model],
            "feature_layers": [feature_layers],
            "backbone": [backbone],
            "weights": [weights],
            "eval_datetime": [pd.Timestamp.now()],
            "bootstrap_layer": [bootstrap_layer],
            "epochs": [epochs],
        }
    )
    if os.path.isfile(output_path):
        old_df = pd.read_csv(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(output_path, index=False)


