import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        ###################################
        # Write your own code here #
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.Dropout(0.1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2)),
            nn.Dropout(0.1),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(115200,2048, bias=True),
            nn.Linear(2048,4, bias=True)
            )

    def forward(self, x):
        out = self.network(x)
        return out