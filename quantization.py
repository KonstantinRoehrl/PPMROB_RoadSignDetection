import os
import sys
import time
import numpy as np
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import config as cfg
import sys
import platform
working_dir = Path(__file__).parent

# from pytorch_quantization import quant_modules
# quant_modules.initialize()

BACKEND = 'fbgemm' if platform.processor() != 'arm64' else 'qnnpack'

# Specify quantization methods in Mode 'enum'
class Mode:
    POST_TRAINING_DYNAMIC_QUANTIZATION = 0  # Apply dynamic quantization
    POST_TRAINING_STATIC_QUANTIZATION = 1   # Apply post static quantization
    QUANTIZATION_AWARE_TRAINING = 2         # Apply quantization aware training
    QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED = 3      # Apply quantization aware training on custom fused model
    
    # Specify paths where models are saved / loaded
    POST_TRAINING_DYNAMIC_QUANTIZATION_PATH = 'dynamic.zip'
    POST_TRAINING_STATIC_QUANTIZATION_PATH = 'static.zip' 
    QUANTIZATION_AWARE_TRAINING_PATH = 'qat.zip'
    QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED_PATH = 'static_fused.zip'

# Loads a model given a model_file path where the state dict is saved.
# An optional model_class parameter can be given to specify the model in which the state dict is loaded
def load_model(model_file, model_class = None):
    if model_class == None:
        model = cfg.model_class(num_classes=4)
    else:
        model = model_class(num_classes=4)
    
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict, strict=False) #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
    model.to(cfg.DEVICE)
    return model

# Loads a quantized model. 
# An optional model_class parameter can be given to specify the model in which the state dict is loaded
def load_quantized_model(
    model = torchvision.models.resnet18(num_classes=4),
    mode: Mode = Mode.QUANTIZATION_AWARE_TRAINING,
    quant_model_file = ''):
    # Quantize untrained model with 'empty' weights to get a quantized model structure
    # and load state dict of previously quantized object.
    #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
    model.to(cfg.DEVICE)
    quant_model = quantize_model(model, mode)
    quant_model.to(cfg.DEVICE)
    state_dict = torch.load(quant_model_file)
    quant_model.load_state_dict(state_dict, strict=False) 

    return quant_model
 
def get_quantized_model_path(mode: Mode):
    file_name = ''
    if mode == Mode.POST_TRAINING_STATIC_QUANTIZATION:
        file_name = Mode.POST_TRAINING_STATIC_QUANTIZATION_PATH
    elif mode == Mode.POST_TRAINING_DYNAMIC_QUANTIZATION:
        file_name = Mode.POST_TRAINING_DYNAMIC_QUANTIZATION_PATH
    elif mode == Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED:
        file_name = Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED_PATH
    elif mode == Mode.QUANTIZATION_AWARE_TRAINING:
        file_name = Mode.QUANTIZATION_AWARE_TRAINING_PATH

    return os.path.join(working_dir, "weights", file_name)
 

def quantize_model(model, mode: Mode = Mode.POST_TRAINING_STATIC_QUANTIZATION):
    if mode == Mode.POST_TRAINING_STATIC_QUANTIZATION:
        print('Quantizing using Post')
        quantized_model = _quantize_model_static(model)
    elif mode == Mode.QUANTIZATION_AWARE_TRAINING:
        quantized_model = _quantize_model_qat(model)
    elif mode == Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED:
        quantized_model = _quantize_fused_model(model)
    elif mode == Mode.POST_TRAINING_DYNAMIC_QUANTIZATION:
        quantized_model = _quantize_model_dynamic(model)
    return quantized_model

def _quantize_model_static(model):
# Post training static quantization
    #https://pytorch.org/tutorials/recipes/quantization.html
    # set quantization config for server (x86)
    model.qconfig = torch.quantization.get_default_qconfig(BACKEND)

    # insert observers
    torch.quantization.prepare(model, inplace=True)
    # Calibrate the model and collect statistics

    # convert to quantized version
    fbgemm_model = torch.quantization.convert(model, inplace=True)
    return fbgemm_model

def _quantize_model_dynamic(model):
    # Post Training Dynamic Quantization
    # https://pytorch.org/tutorials/recipes/quantization.html
    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model, 
        qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    return model_dynamic_quantized

def _quantize_model_qat(model):
    # Quantization aware training
    # https://pytorch.org/tutorials/recipes/quantization.html
    model.qconfig = torch.quantization.get_default_qat_qconfig(BACKEND)
    model_qat = torch.quantization.prepare_qat(model, inplace=False)
    # quantization aware training goes here
    model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)
    return model_qat

# Create custom fused and quantized resnet18 model 
#  Quantization - Add quantization to input / Dequantization to output layer
# Error in running quantised model RuntimeError: Could not run ‘quantized::conv2d.new’ with arguments from the ‘CPU’ backend 
# https://discuss.pytorch.org/t/error-in-running-quantised-model-runtimeerror-could-not-run-quantized-conv2d-new-with-arguments-from-the-cpu-backend/151718/2
# Adding quantization stubs to resnet 18: 
# https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def fuse_resnet18(model):
    torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)  

def _get_fused_resnet_model(model: torchvision.models.resnet18):
    model.eval()
    fuse_resnet18(model)
    quantized_model = QuantizedResNet18(model)
    return quantized_model

def _quantize_fused_model(model):
    fused_model = _get_fused_resnet_model(model)
    fused_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    quantized_model = _quantize_model_static(fused_model)
    return quantized_model


## Main
if __name__ == '__main__':

    model_path = os.path.join(working_dir, 'weights', cfg.model_path)
    print('Start quantization of pre-trained model:')
    print(f'Looking for model: {model_path}')

    if not os.path.exists(model_path):
        print(f'ERROR: Model could not be found: {model_path} ! Exiting.')
        sys.exit(0)
    
    print('Start quantizing')    

    quantization_techniques = [
        Mode.POST_TRAINING_DYNAMIC_QUANTIZATION,
        Mode.POST_TRAINING_STATIC_QUANTIZATION,
        Mode.QUANTIZATION_AWARE_TRAINING,
        Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED
    ]

    for technique in quantization_techniques:
        # Reload model from scratch
        model = load_model(model_path)
        quantized_model = quantize_model(model, technique)
        torch.save(quantized_model.state_dict(), get_quantized_model_path(technique))
        print(f'FINISHED Quantization: {technique}. Saved to: {get_quantized_model_path(technique)}')   


        