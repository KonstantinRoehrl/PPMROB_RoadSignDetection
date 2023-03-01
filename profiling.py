# 
# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
#
#

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from quantization import load_model, load_quantized_model, Mode, get_quantized_model_path
import config as cfg
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision
import os

working_dir = Path(__file__).parent

if __name__ == '__main__':
    # Create 'profiling' folder if not exists
    if not os.path.exists(os.path.join(working_dir, 'profiling')):
        os.mkdir(os.path.join(working_dir, 'profiling'))

    path = get_quantized_model_path(Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED)

    model = load_model(path)
    quantized_model = load_model(path, model_class=torchvision.models.quantization.resnet18)

    profiling_models = [quantized_model, model]
    inputs = torch.randn(500, 3, 128, 128)

    for model in profiling_models:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(inputs)

        profiling_report = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        with open(f"./profiling/{model.__class__}´_profiling.txt", "w") as f:
           print(profiling_report, file=f)
        f.close()

        time.sleep(5.0)