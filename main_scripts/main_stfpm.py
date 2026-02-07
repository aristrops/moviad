import os, sys
from datetime import datetime
import argparse
from glob import glob
import torch
from torch.utils.data import DataLoader

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, "..", ".."))

from moviad.models.stfpm.stfpm import STFPM
from moviad.trainers.trainer_stfpm import TrainerSTFPM
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
#from moviad.trainers.trainer_stfpm import train_param_grid_search
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split


# 1. train stfpm on a category
# 2. save the best model
# 3. load the model (if not trained in the same run)
# 4. run test on the model
# 5. compute the metrics


def main(args):
    dataset_path = args.dataset_path
    results_dirpath = args.results_dirpath
    checkpoint_dir = args.checkpoint_dir
    trained_models_filepaths = args.trained_models_filepaths
    output_size = args.output_size
    batch_size = args.batch_size
    model_name = args.model_name
    ad_layers = args.ad_layers
    categories = args.categories
    boot_layer = args.boot_layer
    seeds = args.seeds
    epochs = args.epochs
    disable_dataset_norm: bool = args.disable_dataset_norm
    feature_maps_dir = args.feature_maps_dir
    contamination_ratio = args.contamination_ratio
    test_dataset = args.test_dataset

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

        ad_layers = [ad_layers]

        params = {
            "dataset_path": dataset_path,
            "categories": categories,
            "ad_layers": ad_layers,
            "epochs": [epochs] * len(ad_layers),
            "seeds": seeds,
            "batch_size": 64,
            "backbone_model_name": model_name,
            "device": device,
            "img_input_size": input_sizes[model_name],
            "img_output_size": output_size,
            "early_stopping": None,  # 0.01 | None | 0.002
            "student_bootstrap_layer": (
                [boot_layer] if boot_layer is not None else [False]
            ),
            "checkpoint_dir": checkpoint_dir,
            "normalize_dataset": bool(not disable_dataset_norm),
            "log_dirpath": args.log_dirpath,
            "contamination_ratio": contamination_ratio,
            "test_dataset": test_dataset
        }

        # input(f"Training with params: {params}\n\nPress Enter to continue...")
        print(f"Training with params: {params}")

        train_dataset = VisaDataset(dataset_path, csv_path=os.path.join(dataset_path, "split_csv", "1cls.csv"),
                                       split=Split.TRAIN, class_name=category)
        train_dataset.load_dataset()
        print(f"Length train dataset: {len(train_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        teacher_model = CustomFeatureExtractor(model_name, ad_layers, device, True, False, None)
        student_model = CustomFeatureExtractor(model_name, ad_layers, device, False, False, None)
        stfpm = STFPM(teacher = teacher_model, student = student_model)

        trainer = TrainerSTFPM(stfpm, train_dataloader, device, checkpoint_dir)
        trained_models_filepaths = trainer.train(params["epochs"][0])

        m = "\n".join(trained_models_filepaths)
        print(f"Trained models:{m}")

    # LOAD AND EVALUATE ###############################################
    if args.eval:
        if trained_models_filepaths is None:
            trained_models_filepaths = glob(
                os.path.join(checkpoint_dir, "**/*.pth.tar"), recursive=True
            )
        if len(trained_models_filepaths) == 0:
            raise ValueError(f"No trained models found in {checkpoint_dir}")

        if not os.path.exists(results_dirpath):
            os.makedirs(results_dirpath)

        models = list(input_sizes.keys())

        for checkpoint_path in trained_models_filepaths:
            torch.manual_seed(0)

            # get category from dirname
            category = os.path.basename(os.path.dirname(checkpoint_path))

            # get backbone model name from filename
            backbone_model_name = [
                m for m in models if m in os.path.basename(checkpoint_path)
            ][0]
            img_input_size = input_sizes[backbone_model_name]

            print(f"backbone model name: {backbone_model_name}")
            print(f"img_input_size: {img_input_size}")
            print(f"Category: {category}")

            test_dataset = MVTecDataset(
                TaskType.SEGMENTATION,
                dataset_path,
                category,
                Split.TEST,
                img_size=img_input_size,
                gt_mask_size=output_size,
                norm=not disable_dataset_norm,
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True
            )
            print(f"Length test dataset: {len(test_dataset)}")

            # load the model snapshot
            model = Stfpm()
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device), strict=False
            )
            model.to(device)

            # evaluate the model
            evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
            scores = evaluator.evaluate(model)

            # save the scores
            metrics_filename = os.path.join(
                results_dirpath,
                f"metrics_{backbone_model_name}.csv",
            )
            append_results(
                metrics_filename,
                category,
                model.seed,
                *scores,
                ad_model,
                str(model.ad_layers),
                backbone_model_name,
                model.weights_name,
                model.student_bootstrap_layer,
                model.epochs,
                img_input_size,
                output_size,
            )

    if args.visualize:
        if trained_models_filepaths is None:
            trained_models_filepaths = glob(
                os.path.join(checkpoint_dir, "**/*.pth.tar"), recursive=True
            )
        if len(trained_models_filepaths) == 0:
            raise ValueError(f"No trained models found in {checkpoint_dir}")

        if not os.path.exists(results_dirpath):
            os.makedirs(results_dirpath)

        models = list(input_sizes.keys())

        models_to_load = []
        for checkpoint_path in trained_models_filepaths:
            if model_name not in checkpoint_path:
                continue
            found = False
            for cat in args.categories:
                if cat in checkpoint_path:
                    found = True
                    break
            if not found:
                continue
            if boot_layer is not None:
                if f"boots{boot_layer}" not in checkpoint_path:
                    continue
            else:
                if "boots" in checkpoint_path:
                    continue
            models_to_load.append(checkpoint_path)

        print("-" * 20, "Models to load", "-" * 20)
        print(models_to_load)

        assert len(models_to_load) > 0, "No models to load"

        for checkpoint_path in models_to_load:
            torch.manual_seed(0)

            # get category from dirname
            category = os.path.basename(os.path.dirname(checkpoint_path))

            # get backbone model name from filename
            backbone_model_name = [
                m for m in models if m in os.path.basename(checkpoint_path)
            ][0]
            img_input_size = input_sizes[backbone_model_name]

            print(f"backbone model name: {backbone_model_name}")
            print(f"img_input_size: {img_input_size}")
            print(f"Category: {category}")

            test_dataset = MVTecDataset(
                TaskType.SEGMENTATION,
                dataset_path,
                category,
                Split.TEST,
                img_size=img_input_size,
                gt_mask_size=output_size,
                normalize=not disable_dataset_norm,
            )
            test_dataset.load_dataset()
            # test_dataloader = DataLoader(
            #     test_dataset, batch_size=batch_size, shuffle=True
            # )
            print(f"Length test dataset: {len(test_dataset)}")

            # load the model snapshot
            model = Stfpm()
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device), strict=False
            )
            model.to(device)

            # save the scores
            visualization_path = os.path.join(
                feature_maps_dir,
                f"{category}_{backbone_model_name}_lay{model.ad_layers}_share{model.student_bootstrap_layer}",
            )
            # 3 files: teacher_maps, student_maps, diff_maps
            teacher_maps, student_maps, diff_maps = [], [], []
            anomaly_maps, original_imgs, labels = [], [], []
            masks = []
            model.attach_hooks(teacher_maps, student_maps)

            for image, label, mask, path in test_dataset:
                model.eval()
                image = image.unsqueeze(0)
                with torch.no_grad():
                    anomaly_map, anomaly_score = model(image.to(device))
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                    original_imgs.append(image.cpu().numpy())
                    labels.append(label)
                    masks.append(mask.cpu().numpy())

            import pickle

            if not os.path.exists(visualization_path):
                os.makedirs(visualization_path)

            print("num inferences: len(teacher_maps)", len(teacher_maps))
            print("len(student_maps)", len(student_maps))

            print("Len of teacher_maps[0]", len(teacher_maps[0]))
            print("Len of student_maps[0]", len(student_maps[0]))

            with open(os.path.join(visualization_path, "teacher_maps.pkl"), "wb") as f:
                pickle.dump(teacher_maps, f)
            with open(os.path.join(visualization_path, "student_maps.pkl"), "wb") as f:
                pickle.dump(student_maps, f)
            with open(os.path.join(visualization_path, "anomaly_maps.pkl"), "wb") as f:
                pickle.dump(anomaly_maps, f)
            with open(os.path.join(visualization_path, "original_imgs.pkl"), "wb") as f:
                pickle.dump(original_imgs, f)
            with open(os.path.join(visualization_path, "labels.pkl"), "wb") as f:
                pickle.dump(labels, f)
            with open(os.path.join(visualization_path, "masks.pkl"), "wb") as f:
                pickle.dump(masks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Backbone model name."
    )
    parser.add_argument(
        "--ad_layers",
        type=int,
        nargs="+",
        help="List of layer indexes to use for feature extraction.",
    )
    parser.add_argument(
        "--boot_layer",
        type=int,
        default=None,
        help="PaSTe will use the specified layer index as the student input.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=CATEGORIES,
        help="Dataset categories to train and evaluate.",
    )
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the model on test set and compute metrics.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="cpu, cuda:0, cuda:1, ..."
    )
    parser.add_argument("--dataset_path", type=str, default="../../datasets/mvtec/")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/stfpm",
        help="Directory to save the trained models.",
    )
    parser.add_argument(
        "--results_dirpath",
        type=str,
        default="./metrics/stfpm",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--log_dirpath",
        type=str,
        default=None,
        help="Directory to save the training logs.",
    )
    parser.add_argument(
        "--trained_models_filepaths",
        type=str,
        default=None,
        nargs="+",
        required=False,
        help="Path to the trained models to use for evaluation only.",
    )
    parser.add_argument("--output_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--input_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds on which to train the model.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--disable_dataset_norm",
        action="store_true",
        help="Disable dataset normalization.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the feature maps."
    )
    parser.add_argument(
        "--feature_maps_dir",
        type=str,
        default=None,
        help="Directory to save the feature maps.",
    )

    parser.add_argument(
        "--contamination_ratio",
        type=float,
        default=None,
        help="Percentage of contamination to introduce in the training set.",
    )

    parser.add_argument(
    "--test_dataset", action="store_true", help="Use test dataset for contamination."
)

    args = parser.parse_args()

    try:
        main(args)

        # create a log file if it does not exist
        if not os.path.exists("stfpm.log"):
            with open("stfpm.log", "w") as f:
                f.write("")
        # write the args as a string to the log file
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("stfpm.log", "a") as f:
            f.write("finished " + "\t" + now_str + "\t" + str(args) + "\n")

    except Exception as e:

        # write the args as a string to the log file
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("stfpm.log", "a") as f:
            f.write("** FAILED **" + "\t" + now_str + "\t" + str(args) + "\n")

        raise e
