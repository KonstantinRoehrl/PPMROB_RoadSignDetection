from pathlib import Path
import os
import torch
import torchvision
import sys

# PATHS
working_dir = Path(__file__).parent
dataset_path = os.path.join(working_dir, "roadSignSet")
model_path = os.path.join(working_dir, "weights", "best_model.zip")

# DEFINE MODEL
#model_class = torchvision.models.AlexNet
model_class = torchvision.models.resnet18

# TRAINING PARAMETERS
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 20

# DEVICE
if sys.platform == 'linux':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif sys.platform == 'darwin':
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
else:
    DEVICE = torch.device('cpu')


# For better Readability
class color:
   PURPLE = '\033[1;35;48m \r'
   CYAN = '\033[1;36;48m \r'
   BOLD = '\033[1;37;48m \r'
   BLUE = '\033[1;34;48m \r'
   GREEN = '\033[1;32;48m \r'
   YELLOW = '\033[1;33;48m \r'
   RED = '\033[1;31;48m \r'
   BLACK = '\033[1;30;48m \r'
   UNDERLINE = '\033[4;37;48m \r'
   END = '\033[1;37;0m'