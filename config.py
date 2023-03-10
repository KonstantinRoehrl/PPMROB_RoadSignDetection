from pathlib import Path
import os
import torch
import torchvision
import sys

# PATHS
working_dir = Path(__file__).parent
dataset_path = os.path.join(working_dir, "roadSignSet")
model_path = os.path.join(working_dir, "weights", "best_model.zip")
model_quant_path = os.path.join(working_dir, "weights", "best_model_quant.zip")

# Create weight folder, if doesn't exist
if not os.path.exists(os.path.join(working_dir, 'weights')):
   os.mkdir(os.path.join(working_dir, 'weights'))

# DEFINE MODEL
# See here [https://pytorch.org/vision/stable/models.html]
#model_class = torchvision.models.AlexNet        # Accuracy 80.75
model_class = torchvision.models.resnet18      # Accuracy 100%
#model_class = torchvision.models.resnet50       # Accuracy 95.4 

# Quantized resnet
#model_class = torchvision.models.quantization.resnet18

# TRAINING PARAMETERS
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 10

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