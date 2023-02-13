import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import cv2
from torchvision.io import read_image
import torchvision.transforms as transforms

import config as cfg
from config import color as col

class UnNormalize(object):
    def __init__(self):
        # Hardcoded for RoadSignSet
        self.mean =  (0.4307480482741943, 0.4665043564842691, 0.6313285444108061)
        self.std =  (0.12267207683226942, 0.03794770827614883, 0.05556710696554743)  

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class RoadSignSet(Dataset):
    def __init__(self, split, dataset_path):
        self.dataset_path = dataset_path
        self.split = split

        # Hardcoded for RoadSignSet
        self.mean =  (0.4307480482741943, 0.4665043564842691, 0.6313285444108061)
        self.std =  (0.12267207683226942, 0.03794770827614883, 0.05556710696554743)  

        if self.split not in ['train', 'test']:
            raise ValueError("Split only allowed to be either 'train' or 'test'.")
        self.samples = self.load_samples()

        size = (128, 128)

        print(col.YELLOW, f"=> Loaded {len(self.samples)} images for {self.split} split.", col.END)

        if self.split == 'train':
            # Augmentations for train split
            self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(size=size),
                                        transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8, 1.2),saturation=(0.8,1.2),hue=(-0.1, 0.1)),
                                        transforms.Normalize(mean=self.mean, std=self.std),
                                        ])
        else: 
            # No color jitter for eval split
            self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(size=size),
                                        transforms.Normalize(mean=self.mean, std=self.std),
                                        ])


    def __len__(self):
        return len(self.samples)

    def load_samples(self):
        samples = []
        # Iterate over categories
        for label in os.listdir(self.dataset_path):
            if label in ['1', '2', '3', '4']:
                label_num = int(label)
                
                category_path = os.path.join(self.dataset_path, label)
                # Iterate over images in one category
                for image in os.listdir(category_path):
                    if image.startswith('.'):
                        # Don't take hidden files
                        continue

                    image_num = int(image.replace(".png", ""))
                    
                    # NOTE: Old datasplit (Every fifth image for datasplit)
                    #if (image_num % 5 == 0 and self.split == 'train') or (image_num % 5 != 0 and self.split == 'test'):
                    #    continue

                    if (self.split == 'train' and image_num < 50) or (image_num >= 50 and self.split == 'test'):
                        continue
                    sample = {
                        'path'  : os.path.join(category_path, image),       # Path to load image
                        'label' : label_num                                 # The corresponding label
                    } 

                    # Add sample to all samples
                    samples.append(sample)
        return samples
                    


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['path'])

        # Convert to yuv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Create one-hot vector
        label = sample['label']
        labels = torch.zeros(4)
        labels[label - 1] = 1

        # Augmentation
        if self.transform != None:
            image = self.transform(image)

        return image.float(), labels