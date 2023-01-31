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


class RoadSignSet(Dataset):
    def __init__(self, split, dataset_path, mean=None, std=None, normalize=True):
        self.dataset_path = dataset_path
        self.split = split
        self.mean = mean
        self.std = std
        if self.split not in ['train', 'test']:
            raise ValueError("Split only allowed to be either 'train' or 'test'.")
        self.samples = self.load_samples()

        size = (256, 256)

        print(col.YELLOW, f"=> Loaded {len(self.samples)} images for {self.split} split.", col.END)

        if normalize and mean != None and std != None:
            self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(size=size),
                                        transforms.Normalize(mean=mean, std=std)
                                        ])
        else:
            self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(size=size)
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
                    
                    # Every fifth image goes to eval split
                    if (image_num % 5 == 0 and self.split == 'train') or (image_num % 5 != 0 and self.split == 'test'):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        label = sample['label']
        labels = torch.zeros(4)
        labels[label - 1] = 1
        if self.transform != None:
            image = self.transform(image)
        return image.float(), labels