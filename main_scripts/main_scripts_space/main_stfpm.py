import os, sys, json, logging
from datetime import datetime
import argparse
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, "..", ".."))

from moviad.models.stfpm.stfpm import Stfpm
from moviad.trainers.trainer_stfpm import train_model
from moviad.datasets.space_datasets import LunarDataset, MarsDataset, compute_mean_std
from moviad.utilities.evaluator import Evaluator


# 1. train stfpm on a category
# 2. save the best model
# 3. load the model (if not trained in the same run)
# 4. run test on the model
# 5. compute the metrics


def main(args):
    dataset_type = args.dataset_type
    output_size = args.output_size
    batch_size = args.batch_size
    model_name = args.model_name
    ad_layers = args.ad_layers
    seeds = args.seeds
    epochs = args.epochs
    save_path = args.save_path
    disable_dataset_norm: bool = args.disable_dataset_norm
    feature_maps_dir = args.feature_maps_dir

    device = torch.device(args.device)
    input_sizes = {
        "mcunet-in3": (176, 176),
        "mobilenet_v2": (224, 224),
        "phinet_1.2_0.5_6_downsampling": (224, 224),
        "wide_resnet50_2": (224, 224),
        "micronet-m1": (224, 224),
    }

    if args.input_size is not None:
        input_sizes[model_name] = args.input_size

    # TRAIN AND SAVE BEST ###############################################
    if args.train:
        ad_model = "stfpm"

        #ad_layers = [ad_layers]

        print(f"Training STFPM for {dataset_type} dataset... \n")

        if dataset_type == "mars":
            mars_mean, mars_std = compute_mean_std(
                MarsDataset(root_dir=r"C:\Users\arist\Downloads\3659202", split="train", transform=None))
            mars_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=mars_mean.tolist(), std=mars_std.tolist()),
            ])

            train_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="train",
                                        transform=mars_transform)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        print(f"Length train dataset: {len(train_dataset)}")

        if dataset_type == "mars":
            test_dataset = MarsDataset(root_dir="vad_space_datasets/mars", split="test",
                                       transform=mars_transform)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
        print(f"Length test dataset: {len(test_dataset)}")

        model = Stfpm(backbone_model_name=model_name, ad_layers=ad_layers)
        model.to(device)

        train_model(model, train_dataloader, test_dataloader, epochs, device, model_save_path=save_path, seed=seeds)

    # # LOAD AND EVALUATE ###############################################
    # if args.eval:
    #     if trained_models_filepaths is None:
    #         trained_models_filepaths = glob(
    #             os.path.join(checkpoint_dir, "**/*.pth.tar"), recursive=True
    #         )
    #     if len(trained_models_filepaths) == 0:
    #         raise ValueError(f"No trained models found in {checkpoint_dir}")
    #
    #     if not os.path.exists(results_dirpath):
    #         os.makedirs(results_dirpath)
    #
    #     models = list(input_sizes.keys())
    #
    #     for checkpoint_path in trained_models_filepaths:
    #         torch.manual_seed(0)
    #
    #         # get category from dirname
    #         category = os.path.basename(os.path.dirname(checkpoint_path))
    #
    #         # get backbone model name from filename
    #         backbone_model_name = [
    #             m for m in models if m in os.path.basename(checkpoint_path)
    #         ][0]
    #         img_input_size = input_sizes[backbone_model_name]
    #
    #         print(f"backbone model name: {backbone_model_name}")
    #         print(f"img_input_size: {img_input_size}")
    #         print(f"Category: {category}")
    #
    #         test_dataset = MVTecDataset(
    #             TaskType.SEGMENTATION,
    #             dataset_path,
    #             category,
    #             Split.TEST,
    #             img_size=img_input_size,
    #             gt_mask_size=output_size,
    #             normalize=not disable_dataset_norm,
    #         )
    #         test_dataloader = DataLoader(
    #             test_dataset, batch_size=batch_size, shuffle=True
    #         )
    #         print(f"Length test dataset: {len(test_dataset)}")
    #
    #         # load the model snapshot
    #         model = Stfpm()
    #         model.load_state_dict(
    #             torch.load(checkpoint_path, map_location=device), strict=False
    #         )
    #         model.to(device)
    #
    #         # evaluate the model
    #         evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
    #         scores = evaluator.evaluate(model)
    #
    #         # save the scores
    #         metrics_filename = os.path.join(
    #             results_dirpath,
    #             f"metrics_{backbone_model_name}.csv",
    #         )
    #         append_results(
    #             metrics_filename,
    #             category,
    #             model.seed,
    #             *scores,
    #             ad_model,
    #             str(model.ad_layers),
    #             backbone_model_name,
    #             model.weights_name,
    #             model.student_bootstrap_layer,
    #             model.epochs,
    #             img_input_size,
    #             output_size,
    #         )
    #
    # if args.visualize:
    #     if trained_models_filepaths is None:
    #         trained_models_filepaths = glob(
    #             os.path.join(checkpoint_dir, "**/*.pth.tar"), recursive=True
    #         )
    #     if len(trained_models_filepaths) == 0:
    #         raise ValueError(f"No trained models found in {checkpoint_dir}")
    #
    #     if not os.path.exists(results_dirpath):
    #         os.makedirs(results_dirpath)
    #
    #     models = list(input_sizes.keys())
    #
    #     models_to_load = []
    #     for checkpoint_path in trained_models_filepaths:
    #         if model_name not in checkpoint_path:
    #             continue
    #         found = False
    #         for cat in args.categories:
    #             if cat in checkpoint_path:
    #                 found = True
    #                 break
    #         if not found:
    #             continue
    #         if boot_layer is not None:
    #             if f"boots{boot_layer}" not in checkpoint_path:
    #                 continue
    #         else:
    #             if "boots" in checkpoint_path:
    #                 continue
    #         models_to_load.append(checkpoint_path)
    #
    #     print("-" * 20, "Models to load", "-" * 20)
    #     print(models_to_load)
    #
    #     assert len(models_to_load) > 0, "No models to load"
    #
    #     for checkpoint_path in models_to_load:
    #         torch.manual_seed(0)
    #
    #         # get category from dirname
    #         category = os.path.basename(os.path.dirname(checkpoint_path))
    #
    #         # get backbone model name from filename
    #         backbone_model_name = [
    #             m for m in models if m in os.path.basename(checkpoint_path)
    #         ][0]
    #         img_input_size = input_sizes[backbone_model_name]
    #
    #         print(f"backbone model name: {backbone_model_name}")
    #         print(f"img_input_size: {img_input_size}")
    #         print(f"Category: {category}")
    #
    #         test_dataset = MVTecDataset(
    #             TaskType.SEGMENTATION,
    #             dataset_path,
    #             category,
    #             Split.TEST,
    #             img_size=img_input_size,
    #             gt_mask_size=output_size,
    #             normalize=not disable_dataset_norm,
    #         )
    #         # test_dataloader = DataLoader(
    #         #     test_dataset, batch_size=batch_size, shuffle=True
    #         # )
    #         print(f"Length test dataset: {len(test_dataset)}")
    #
    #         # load the model snapshot
    #         model = Stfpm()
    #         model.load_state_dict(
    #             torch.load(checkpoint_path, map_location=device), strict=False
    #         )
    #         model.to(device)
    #
    #         # save the scores
    #         visualization_path = os.path.join(
    #             feature_maps_dir,
    #             f"{category}_{backbone_model_name}_lay{model.ad_layers}_share{model.student_bootstrap_layer}",
    #         )
    #         # 3 files: teacher_maps, student_maps, diff_maps
    #         teacher_maps, student_maps, diff_maps = [], [], []
    #         anomaly_maps, original_imgs, labels = [], [], []
    #         masks = []
    #         model.attach_hooks(teacher_maps, student_maps)
    #
    #         for image, label, mask, path in test_dataset:
    #             model.eval()
    #             image = image.unsqueeze(0)
    #             with torch.no_grad():
    #                 anomaly_map, anomaly_score = model(image.to(device))
    #                 anomaly_maps.append(anomaly_map.cpu().numpy())
    #                 original_imgs.append(image.cpu().numpy())
    #                 labels.append(label)
    #                 masks.append(mask.cpu().numpy())
    #
    #         import pickle
    #
    #         if not os.path.exists(visualization_path):
    #             os.makedirs(visualization_path)
    #
    #         print("num inferences: len(teacher_maps)", len(teacher_maps))
    #         print("len(student_maps)", len(student_maps))
    #
    #         print("Len of teacher_maps[0]", len(teacher_maps[0]))
    #         print("Len of student_maps[0]", len(student_maps[0]))
    #
    #         with open(os.path.join(visualization_path, "teacher_maps.pkl"), "wb") as f:
    #             pickle.dump(teacher_maps, f)
    #         with open(os.path.join(visualization_path, "student_maps.pkl"), "wb") as f:
    #             pickle.dump(student_maps, f)
    #         with open(os.path.join(visualization_path, "anomaly_maps.pkl"), "wb") as f:
    #             pickle.dump(anomaly_maps, f)
    #         with open(os.path.join(visualization_path, "original_imgs.pkl"), "wb") as f:
    #             pickle.dump(original_imgs, f)
    #         with open(os.path.join(visualization_path, "labels.pkl"), "wb") as f:
    #             pickle.dump(labels, f)
    #         with open(os.path.join(visualization_path, "masks.pkl"), "wb") as f:
    #             pickle.dump(masks, f)


if __name__ == "__main__":
    # try:
    #     paths = json.load(open("../../utilities/pathconfig.json"))
    # except FileNotFoundError:
    #     logging.warning("Could not load pathconfig.json. Make sure the file exists.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--ad_layers", type=int, nargs="+")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--output_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--input_size", type=int, nargs=2, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--disable_dataset_norm", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--feature_maps_dir", type=str, default=None)

    args = parser.parse_args()

    main(args)