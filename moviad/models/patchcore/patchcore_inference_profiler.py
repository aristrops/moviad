import os
import time
import psutil
import random
from pathlib import Path

from memory_profiler import memory_usage
from calflops import calculate_flops

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.patchcore.patchcore import PatchCore
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.get_sizes import get_tensor_size
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)


def static_profile(model, img_size, batch_size, device, quantize_mb):
    print("--- Static Profiling ---")

    # GPU backbone FLOPs
    input_shape = (batch_size, 3, img_size[0], img_size[1])

    model = model.to(device)
    model.eval()

    flops, macs, _ = calculate_flops(
        model=model.feature_extractor.model,
        input_shape=tuple(input_shape),
        output_as_string=False,
        print_results=False,
        output_precision=4,
    )

    # number of parameters (frozen)
    total_params = sum(p.numel() for p in model.feature_extractor.model.parameters())

    print(f"Backbone Params: {total_params} M")
    print(f"Backbone MACs (per batch): {macs / 1e9:.3f} G")
    print(f"Backbone FLOPs (per batch): {flops / 1e9:.3f} G")

    # CPU model static memory
    bytes_per_elem = 4
    
    # infer output shape
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model.feature_extractor.model(dummy_input)

    # MB size
    mb_size = get_tensor_size(model.memory_bank)
    #add codebook size if quantized
    pq_codebooks_size = model.product_quantizer.get_size_mb() if quantize_mb else 0

    print("\nCPU static memory:")
    print(f"Memory bank: {mb_size / 1e6:.2f} MB")
    print(f"Codebooks size: {pq_codebooks_size / 1e6:.2f} MB")
    print(f"Total stats: {(mb_size + pq_codebooks_size) / 1e6:.2f} MB")

    # CPU distance computation FLOPs
    # per_loc_flops = None

    # if not model.diag_cov:
    #     per_loc_flops = feature_dim**2 + 2 * feature_dim
    # else:
    #     per_loc_flops = feature_dim + 2 * feature_dim

    # per_img_flops = spatial_dim * per_loc_flops
    # per_batch_flops = batch_size * per_img_flops

    # print("\nCPU distance computation FLOPs:")
    # print(f"Per image: {per_img_flops / 1e6:.2f} MFLOPs")
    # print(f"Per batch: {per_batch_flops / 1e6:.2f} MFLOPs")

    return {
        "backbone_flops": flops,
        "backbone_macs": macs,
        "backbone_params": total_params,
        # "cpu_flops_per_image": per_img_flops,
        # "cpu_mem_bytes": mean_mem + cov_mem,
    }


def run_forward(model, images):
    with torch.no_grad():
        return model(images)


def dynamic_profile(model, dataloader, device, num_batches=10):
    """
    Profile VAD model dynamically over the dataloader batches. 
    It measures latency per sample, GPU max memory peak and 
    deltas of the CPU memory. 
    
    Args:
        model: loaded VAD model
        dataloader: dataLoader for the profiling
        device: model device
        num_batches: number of batches to profile
    """
    
    latencies = []
    cpu_mem_peaks = []
    gpu_mem_peak = 0

    # warm-up GPU and CPU
    images = next(iter(dataloader))[0].to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(images)

    # loop over batches
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch[0].to(device)

        # reset GPU stats if device is GPU
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats(device)

            # forward pass
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, _ = model(images)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
        
        else:  # CPU
            with torch.no_grad():
                t0 = time.perf_counter()
                _, _ = model(images)
                t1 = time.perf_counter()

        # forward pass again for CPU peak memory measure
        mem_trace = memory_usage(
            (run_forward, (model, images)),
            interval=0.001,      # 1 ms sampling
            max_iterations=1,
            retval=False
        )

        # latency per sample
        latencies.append((t1 - t0) / images.size(0))

        # GPU memory peak
        if "cuda" in device:
            gpu_mem_peak = max(gpu_mem_peak, torch.cuda.max_memory_allocated(device))

        # CPU memory delta
        cpu_mem_peaks.append(max(mem_trace) - min(mem_trace))

    # report results
    print("\n--- Profiling Results ---")
    print(f"Average latency per sample: {sum(latencies)/len(latencies)*1000:.2f} ms")
    print(f"Max CPU memory delta per batch: {max(cpu_mem_peaks):.2f} MB")
    if "cuda" in device:
        print(f"Max GPU memory usage: {gpu_mem_peak/1024**2:.2f} MB")

    return {
        "latencies_ms": [l*1000 for l in latencies],
        "cpu_mem_peaks_mb": cpu_mem_peaks,
        "gpu_mem_peak_mb": gpu_mem_peak / 1024**2,
    }


def main(args):
    batch_size = args.batch_size  # 32
    save_path = args.save_path  # "output/padim/"
    quantizer_save_path = args.quantizer_save_path if args.quantize_mb else None  
    data_path = args.data_path  # "../datasets/mvtec/"
    device = args.device  # "cuda:1"  # cuda:0, cuda:1, cuda:2, cpu
    backbone_model_name = args.backbone_model_name  # "resnet18"
    #save_figures = args.save_figures  # False
    #results_dirpath = args.results_dirpath  # "metrics/padim/"
    categories = args.categories
    seeds = args.seeds
    img_input_size = args.img_input_size
    #output_size = args.output_size
    ad_layers_idxs = args.ad_layers_idxs
    quantize_mb = args.quantize_mb  # whether the memory bank is quantized

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed_all(seed)

        for category_name in categories:

            print(
                "class name:",
                category_name,
            )

            print("---- PatchCore Profiler ----")

            # load the model if it was not trained in this run
            feature_extractor = CustomFeatureExtractor(backbone_model_name, ad_layers_idxs, device, True, False, None)
            patchcore = PatchCore(device = device,
                                  input_size=img_input_size,
                                  feature_extractor = feature_extractor,
                                  apply_quantization=quantize_mb)
            #path = padim.get_model_savepath(save_path)
            patchcore.load(model_state_dict_path=save_path, quantizer_state_dict_path = quantizer_save_path)
            patchcore.to(device)
            print(f"Loaded model from path: {save_path}")

            patchcore.eval()

            #-------TRAIN PATCHCORE------
           #patchcore.train()

            # train_dataset = MVTecDataset(
            #     TaskType.SEGMENTATION,
            #     data_path,
            #     category_name,
            #     Split.TRAIN,
            #     img_size=img_input_size,
            # )

            # train_dataset.load_dataset()

            # train_dataloader = DataLoader(
            #     train_dataset, batch_size=batch_size, pin_memory=True
            # )

            
            test_dataset = MVTecDataset(
                TaskType.SEGMENTATION,
                data_path,
                category_name,
                Split.TEST,
                img_size=img_input_size,
            )

            test_dataset.load_dataset()

            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, pin_memory=True
            )


            # trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device, evaluate=False)
            # trainer.train()

            #------EVALUATE AND PROFILE PATCHCORE------

            # static profile
            #static_profile(model=patchcore, img_size=(224,224), batch_size=batch_size, device=device, quantize_mb=quantize_mb)
            
            # dynamic profile
            dynamic_profile(model=patchcore, dataloader=test_dataloader, device=device, num_batches=1000)


if __name__ == "__main__":
    import argparse

    categories = [
        "hazelnut",  # at the top because very large in memory, so we can check if it crashes
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--save_figures", action="store_true")
    parser.add_argument("--save_logs", action="store_true")
    parser.add_argument(
        "--backbone_model_name",
        type=str,
        help="resnet18, wide_resnet50_2, mobilenet_v2, mcunet-in3",
    )
    parser.add_argument(
        "--img_input_size",
        type=int,
        default=(224, 224),
        help="input image size, if None, default is used",
    )
    #parser.add_argument(
    #    "--output_size",
    #    type=int,
    #    default=(224, 224),
    #    help="output image size, if None, default is used",
    #)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_path", type=str, default=None, help="where to save the model checkpoint"
    )
    parser.add_argument(
        "--quantizer_save_path", type=str, default=None, help="where to save the product quantizer checkpoint"
    )
    parser.add_argument("--data_path", type=str, default="../../datasets/mvtec/")
    parser.add_argument("--device", type=str, default="cuda:1")
    #parser.add_argument("--results_dirpath", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--categories", type=str, nargs="+", default=categories)
    parser.add_argument(
        "--ad_layers_idxs",
        type=str,
        nargs="+",
        required = True,
        help="list of layers idxs to use for feature extraction",
    )
    parser.add_argument("--quantize_mb", action="store_true", help="whether to quantize memory bank")

    args = parser.parse_args()

    log_filename = "patchcore.log"
    s = "DEBUG " if args.debug else ""

    main(args)