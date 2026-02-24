from tqdm import *
import copy

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moviad.models.stfpm.stfpm import Stfpm
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer
#
# class TrainerSTFPM(Trainer):
#
#     """
#     This class contains the code for training the STFPM model
#     """
#
#     @staticmethod
#     def stfpm_loss(teacher_features, student_features):
#         return torch.sum((teacher_features - student_features) ** 2, 1).mean()
#
#     def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):
#
#         optimizer = torch.optim.SGD(
#             self.model.student.model.parameters(),
#             STFPM.DEFAULT_PARAMETERS["learning_rate"],
#             momentum=STFPM.DEFAULT_PARAMETERS["momentum"],
#             weight_decay=STFPM.DEFAULT_PARAMETERS["weight_decay"]
#         )
#
#         best_metrics = {}
#         best_metrics["img_roc_auc"] = 0
#         best_metrics["pxl_roc_auc"] = 0
#         best_metrics["img_f1"] = 0
#         best_metrics["pxl_f1"] = 0
#         best_metrics["img_pr_auc"] = 0
#         best_metrics["pxl_pr_auc"] = 0
#         best_metrics["pxl_au_pro"] = 0
#
#         # log the training configurations
#         if self.logger:
#             self.logger.config.update(
#                 {
#                     "epochs": epochs,
#                     "learning_rate": STFPM.DEFAULT_PARAMETERS["learning_rate"],
#                     "weight_decay":STFPM.DEFAULT_PARAMETERS["weight_decay"],
#                     "optimizer": "SGD",
#                     "momentum": STFPM.DEFAULT_PARAMETERS["momentum"],
#                 },
#                 allow_val_change=True
#             )
#             self.logger.watch(self.model, log='all', log_freq=10)
#
#         for epoch in trange(epochs):
#
#             self.model.train()
#
#             print(f"EPOCH: {epoch}")
#
#             avg_batch_loss = 0
#             #train the model
#             for batch in tqdm(self.train_dataloader):
#
#                 batch = batch.to(self.device)
#                 teacher_features, student_features = self.model(batch)
#
#                 for i in range(len(student_features)):
#
#                     teacher_features[i] = F.normalize(teacher_features[i], dim=1)
#                     student_features[i] = F.normalize(student_features[i], dim=1)
#                     loss = TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])
#
#                 avg_batch_loss += loss.item()
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             avg_batch_loss /= len(self.train_dataloader)
#             if self.logger:
#                 self.logger.log({
#                     "current_epoch" : epoch,
#                     "avg_batch_loss": avg_batch_loss
#                 })
#
#             if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
#                 print("Evaluating model...")
#                 metrics = self.evaluator.evaluate(self.model)
#
#                 if self.saving_criteria(best_metrics, metrics) and self.save_path is not None:
#                     print("Saving model...")
#                     torch.save(self.model.state_dict(), self.save_path)
#                     print(f"Model saved to {self.save_path}")
#
#                 # update the best metrics
#                 best_metrics = Trainer.update_best_metrics(best_metrics, metrics)
#
#                 print("Trainer training performances:")
#                 Trainer.print_metrics(metrics)
#
#                 if self.logger is not None:
#                     self.logger.log(best_metrics)
#
#         print("Best training performances:")
#         Trainer.print_metrics(best_metrics)
#
#         if self.logger is not None:
#             self.logger.log(
#                 best_metrics
#             )
#
#         best_results = TrainerResult(
#             **best_metrics
#         )
#
#         results = TrainerResult(
#             **metrics
#         )
#
#
#         return results, best_results


import os, json, sys, time, datetime
from typing import Union

from glob import glob
from tqdm import trange
import pandas as pd, numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from moviad.models.stfpm.stfpm import Stfpm, MVTecDataset as StfpmMVTecDataset

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, "..", ".."))

#from datasets.mvtec_dataset import MVTecDataset

# from utilities.evaluator import StfpmEvaluator
from moviad.utilities.configurations import TaskType, Split

#paths = json.load(open("../../utilities/pathconfig.json"))

# dataset_path = paths["mv_tec"]
# log_dirpath = "logs"


def get_normalization_transform(img_input_size, normalize=True):
    """
    Return a transform object that resizes the input image to the given size and normalizes it
    according to the ImageNet mean and standard deviation.
    """
    t = []
    t.append(transforms.Resize(img_input_size))
    t.append(transforms.ToTensor())
    if normalize:
        t.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(t)


def load_datasets(dataset_path, category, img_input_size, batch_size, normalize=True):
    transform = get_normalization_transform(img_input_size, normalize)

    # -- train and validation --
    p = os.path.join(dataset_path, category, "train", "good", "*.png")
    image_list = sorted(glob(p))
    assert len(image_list) > 0, f"No images found for category {category} in path {p}"
    train_image_list, val_image_list = train_test_split(
        image_list, test_size=0.2, random_state=0
    )
    train_dataset = StfpmMVTecDataset(train_image_list, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_dataset = StfpmMVTecDataset(val_image_list, transform=transform)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    # -- test --
    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        dataset_path,
        category,
        Split.TEST,
        img_size=img_input_size,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def save_logs(logs, category, log_dirpath, log_filename):
    df = pd.DataFrame(logs)
    dirpath = os.path.join(log_dirpath, category)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    fp = os.path.join(dirpath, log_filename)
    # if the file already exists, append the new logs
    if os.path.exists(fp):
        df = pd.concat([pd.read_csv(fp), df], ignore_index=True)
    df.to_csv(os.path.join(dirpath, log_filename), index=False)


def train_model(
        model: Stfpm,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        model_save_path: str,
        log_dirpath=None,
        seed=None,
        early_stopping: Union[float, bool] = False,
):
    """
    Train the student-teacher feature-pyramid model and save checkpoints
    for each category.

    Args:
        model: stfpm model
        train_loader: torch dataloader for the training dataset
        val_loader: val_loader: torch dataloader for the validation dataset
        epochs: number of epochs to train the model
        device: where to run the model
        category: name of the mvtec category where to save the model
        model_save_path: directory where to create the category subdirectory
            and save the model
        log_dirpath: directory where to save the training logs
        seed: seed for reproducibility
        early_stopping: if a float is provided, the training will stop if the validation
            loss difference between the current and the previous epoch is less than the
            provided value.
    """
    model.seed = seed
    model.epochs = epochs

    if model.seed is not None:
        torch.manual_seed(model.seed)

    min_err = 10000
    prev_val_loss = 100000

    if "micronet" in model.student.model_name:
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.04, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4
        )

    # simple loss function of STFPM
    def loss_fn(t_feat, s_feat):
        return torch.sum((t_feat - s_feat) ** 2, 1).mean()

    logs = []
    for epoch in trange(epochs, desc="Train stfpm"):
        model.train()
        mean_loss = 0

        # train the model
        for batch_data in tqdm(train_loader):
            # the original loader returns a tuple of two lists, one contains the paths
            # and the other the images
            if isinstance(batch_data, (tuple, list)):
                inputs = batch_data[0]
            else:
                inputs = batch_data

            inputs = inputs.to(device)
            t_feat, s_feat = model(inputs)

            loss = loss_fn(t_feat[0], s_feat[0])
            for i in range(1, len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += loss_fn(t_feat[i], s_feat[i])

            #print("[%d/%d] loss: %f" % (epoch, epochs, loss.item()))
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss /= len(train_loader)

        # evaluate the model
        with torch.no_grad():
            model.eval()
            val_loss = torch.zeros(1, device=device)
            for batch_data in val_loader:
                if isinstance(batch_data, (tuple, list)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data

                inputs = inputs.to(device)
                anomaly_maps, _ = model(inputs.to(device))
                val_loss += anomaly_maps.mean()
            val_loss /= len(val_loader)

        log_dict = {
            "epochs": epoch,
            "val_loss": val_loss.cpu(),
            "train_loss": mean_loss,
        }
        logs.append(log_dict)

        # save best checkpoint
        if val_loss < min_err:
            min_err = val_loss

            # model_filename = model.model_filename()
            # save_path = os.path.join(model_save_path, model.category, model_filename)
            # dir_name = os.path.dirname(save_path)
            # if dir_name and not os.path.exists(dir_name):
            #     os.makedirs(dir_name)
            torch.save(model.state_dict(), model_save_path)

        if early_stopping not in [False, None] and epoch > 0:
            if np.abs(val_loss.cpu() - prev_val_loss) < early_stopping:
                print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                break
        prev_val_loss = val_loss.cpu()

    # logs_df = pd.DataFrame(logs)
    # if log_dirpath is not None:
    #     assert model.category is not None
    #     logs_path = os.path.join(
    #         log_dirpath,
    #         model.category,
    #         "train_logs",
    #         model.model_filename() + "_train_logs.csv",
    #     )
    #     dirpath = os.path.dirname(logs_path)
    #     if not os.path.exists(dirpath):
    #         os.makedirs(dirpath)
    #     logs_df.to_csv(logs_path, index=False)
    #
    # return logs_df, save_path


def train_param_grid_step(
        config,
        batch_size,
        backbone_model_name,
        device,
        img_input_size,
        img_output_size,
        early_stopping=False,
        checkpoint_dir="./snapshots",
        normalize_dataset=True,
):
    category = config["category"]
    ad_layers = config["ad_layers"]
    student_bootstrap_layer = config.get("student_bootstrap_layer", None)
    epochs = config["epochs"]
    seed = config.get("seed", None)

    print(
        f"TRAIN | cat: {category}, ad_layers: {ad_layers}, epochs: {epochs}, seed: {seed}, early_stopping: {early_stopping}, bootstrap: {student_bootstrap_layer}"
    )

    if seed is not None:
        torch.manual_seed(seed)

    start_time = time.time()

    train_loader, val_loader, test_loader = load_datasets(
        dataset_path, category, img_input_size, batch_size, normalize_dataset
    )

    model = Stfpm(
        input_size=img_input_size,
        output_size=img_output_size,
        ad_layers=ad_layers,
        backbone_model_name=backbone_model_name,
        student_bootstrap_layer=student_bootstrap_layer,
    )
    model.to(device)

    train_logs, snapshot_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        category=category,
        model_save_path=checkpoint_dir,
        log_dirpath=log_dirpath,
        seed=seed,
        early_stopping=early_stopping,
    )

    train_time = time.time() - start_time

    # evaluator = StfpmEvaluator(model)
    # image_auc, pixel_auc = evaluator.evaluate_stfpm(test_loader, output_path="")
    image_auc, pixel_auc = -1, -1

    return {
        "image_auc": image_auc,
        "pixel_auc": pixel_auc,
        "train_time": train_time,
        "stop_epoch": train_logs["epochs"].max() + 1,
    }, snapshot_path


default_params = {
    "categories": ["hazelnut"],
    "ad_layers": [[8, 9], [10, 11, 12]],
    "epochs": [3, 3],
    "seeds": [0, 1],
    "batch_size": 32,
    "backbone_model_name": "mobilenet_v2",
    "device": "cuda:2",
    "img_input_size": (224, 224),
    "checkpoint_dir": "snapshots",
    "student_bootstrap_layer": [6, None],
}


def train_param_grid_search(params=default_params):
    """
    Parameters:
        categories: list of categories to train the model on
        ad_layers: N list of lists of integers, each list represents the layers to be used for the AD module
        epochs: N list of integers, each integer represents the number of epochs to train the model
        seeds: list of integers, each integer represents the seed for reproducibility
        student_bootstrap_layer: N list of integers, each integer represents the layer to be used for bootstrapping

    Note:
        ad_layers, epochs, student_bootstrap_layer are lists of the same length
    """
    trained_models_filepaths = []
    log_filename = f"logs_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    boot_layers = params.get(
        "student_bootstrap_layer", [None] * len(params["ad_layers"])
    )
    for category in params["categories"]:
        for ad_layers, epochs, boot_layer in zip(
                params["ad_layers"], params["epochs"], boot_layers
        ):
            logs = []
            for seed in params["seeds"]:
                for boot_layer in params.get("student_bootstrap_layer", (None,)):
                    # config params are the parameters that change from run to run
                    config = {
                        "category": category,
                        "ad_layers": ad_layers,
                        "epochs": epochs,
                        "seed": seed,
                        "batch_size": params["batch_size"],
                        "student_bootstrap_layer": boot_layer,
                    }
                    log, snapshot_path = train_param_grid_step(
                        config,
                        params["batch_size"],
                        params["backbone_model_name"],
                        params["device"],
                        params["img_input_size"],
                        params["img_output_size"],
                        params["early_stopping"],
                        params["checkpoint_dir"],
                        params["normalize_dataset"],
                    )

                    # -- LOGGING --
                    # - config: parameters that are changed from run to run
                    # - log: results of the run
                    # - other parameters that are constant for all runs
                    logs.append(
                        {
                            **config,
                            **log,
                            "backbone_model_name": params["backbone_model_name"],
                            "img_input_size": params["img_input_size"],
                        }
                    )
                    trained_models_filepaths.append(snapshot_path)

            save_logs(logs, category, log_dirpath, log_filename)

    return trained_models_filepaths


if __name__ == "__main__":
    train_param_grid_search()