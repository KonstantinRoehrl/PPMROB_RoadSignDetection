import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import os
from tqdm import tqdm
from drawnow import drawnow
import torchvision
import torch.nn as nn

from dataloader import RoadSignSet, UnNormalize
import config as cfg
from config import color as col
from quantization import get_quantized_model_path, Mode
from pytorch_quantization import quant_modules
quant_modules.initialize()

def load_quant_model(quant_model_file):
    model = cfg.model_class(num_classes=4)
    #model.to(cfg.DEVICE)

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_qat = torch.quantization.prepare_qat(model, inplace=False)
    # quantization aware training goes here
    model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)


    state_dict = torch.load(quant_model_file)
    model_qat.load_state_dict(state_dict, strict=False) #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
    model_qat.to(cfg.DEVICE)
    return model_qat

def load_quant_model_torch_class(quant_model_file):
    model = torchvision.models.quantization.resnet18(num_classes=4)
    model.to(cfg.DEVICE)

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_qat = torch.quantization.prepare(model, inplace=False)
    #quantization aware training goes here
    model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)
    model_qat.to(cfg.DEVICE)

    state_dict = torch.load(quant_model_file)
    model.load_state_dict(state_dict, strict=False) #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
   
    return model 

####
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

def load_quant_model_fused_class(quant_model_file):
  model = torchvision.models.resnet18()
  model.eval()
  fuse_resnet18(model)
  quantized_model = QuantizedResNet18(model)

  quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

  # insert observers
  torch.quantization.prepare(quantized_model, inplace=True)
  # Calibrate the model and collect statistics

  # convert to quantized version
  fbgemm_model = torch.quantization.convert(quantized_model, inplace=True)
  
  state_dict = torch.load(quant_model_file)
  fbgemm_model.load_state_dict(state_dict, strict=False) #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
  fbgemm_model.to(cfg.DEVICE)
  return fbgemm_model
   

def predict(test_loader, model):
    unnorm = UnNormalize()
    progress = tqdm(test_loader)
    progress.set_description("Step through images by pressing a button")

    for data, labels in progress:
      # Data -> Device
      data = data.to(cfg.DEVICE)
      labels = labels.to(cfg.DEVICE)

      # Predict
      outputs = model(data)
      outputs = torch.softmax(outputs, dim=-1)

      pred_index = outputs.argmax(dim=-1)
      
      # Plot
      classes = {0:'Stop Sign',1:'50 Speed Limit',2:'Turn Right',3:'Turn Left'}
      rows, cols = int(data.shape[0]/2), int(data.shape[0]/2)
      ax = []
      for k, image in enumerate(data):
        ax.append(plt.subplot(rows, cols, k+1))
        title = f"Pred: {classes[pred_index[k].item()]} ({outputs[k][pred_index[k].item()]:.2f})"
        ax[-1].set_title(title)

        # Image: unnorm -> cpu -> yuv2rgb -> set range(0, 255)
        img_np = unnorm(image).permute(1, 2, 0).detach().cpu().numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_YUV2RGB)
        img_np = (255 * (img_np/np.max(img_np))).astype(np.uint8)
        plt.imshow(img_np)
      plt.pause(0.01)
      plt.waitforbuttonpress()

  
if __name__ == '__main__':
    # Test data
    test_set = RoadSignSet(split='test', dataset_path=cfg.dataset_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

    #model = cfg.model_class(num_classes=4)
    #model.to(cfg.DEVICE)

    if os.path.exists(cfg.model_quant_path):
        path = get_quantized_model_path(Mode.QUANTIZATION_AWARE_TRAINING_QUANTIZATION_FUSED)
        print(path)
        
        model = load_quant_model_torch_class(path)
        #model = load_quant_model_torch_class(cfg.model_quant_path)
        #model.load_state_dict(torch.load(cfg.model_quant_path))
    else:
        raise FileNotFoundError("No weights found! Please run training first.")

    model.eval()
    with torch.set_grad_enabled(False):
      predict(test_loader, model)