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

from dataloader import RoadSignSet, UnNormalize
import config as cfg
from config import color as col

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

    model = cfg.model_class(num_classes=4)
    model.to(cfg.DEVICE)

    if os.path.exists(cfg.model_path):
        model.load_state_dict(torch.load(cfg.model_path), strict=False)
    else:
        raise FileNotFoundError("No weights found! Please run training first.")

    model.eval()
    with torch.set_grad_enabled(False):
      predict(test_loader, model)