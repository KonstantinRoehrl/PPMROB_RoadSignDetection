import torch
torch.manual_seed(0)  # Deterministic training results
import torch.optim as optim
import numpy as np
import torchvision
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys

from dataloader import RoadSignSet
import config as cfg
from config import color as col

def calculate_mean_std(dataset: torch.utils.data.Dataset, recalculate=False):
  if(recalculate):
    # Generating shape for np.array
    shape = (len(dataset),) + tuple(dataset[0][0].numpy().shape)
    col_channel = np.zeros(shape)

    # Extracting all the image data
    for i, (image, label) in enumerate(dataset):
      col_channel[i][0] = image[0].numpy()
      col_channel[i][1] = image[1].numpy()
      col_channel[i][2] = image[2].numpy()

    mean = (np.mean(col_channel[:,0]),np.mean(col_channel[:,1]),np.mean(col_channel[:,2]))
    std = (np.std(col_channel[:,0]),np.std(col_channel[:,1]),np.std(col_channel[:,2]))

    # print in order to hardcode data
    print("mean = ", mean)
    print("std = ", std)

  else:
    # Hardcoded for RoadSignSet
    mean =  (0.4307480482741943, 0.4665043564842691, 0.6313285444108061)
    std =  (0.12267207683226942, 0.03794770827614883, 0.05556710696554743)  

  ###################################
  return mean, std

def eval(data_loader, model):
    model.eval()
    correct = 0
    total = 0
    progress = tqdm(data_loader)
    progress.set_description("Evaluating")
    y_true = np.zeros(0)
    y_pred = np.zeros(0)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in progress:
            images, labels = data
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            # calculate outputs by running images through the network
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)

            # the class with the highest confidence is what we choose as prediction
            _, predicted = torch.max(outputs, 1)

            total += labels.shape[0]
            _, gt = torch.max(labels, 1)

            y_true = np.append(y_true, gt.cpu().numpy())
            y_pred = np.append(y_pred, predicted.cpu().numpy())

            correct += torch.count_nonzero(gt == predicted)
            accuracy = 100. * (correct / total)
    cf_matrix = confusion_matrix(y_true.tolist(), y_pred.tolist(), labels=[0, 1, 2, 3])
    print("Confusion matrix after Evaluating:\n", cf_matrix)
    accuracy = 100. * (correct / total) # Just take percentage of correct predictions
    return accuracy.item()

def train(data_loader, model, optimizer, criterion):
    model.train()
    progress = tqdm(data_loader)
    progress.set_description("Train Network")
    for k, data in enumerate(progress):

        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(images)

        outputs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

if __name__ == '__main__':

    # For normalizing, we need mean and std. This is hardcoded already.
    # If you want to recalculate it, uncomment this code and then paste it into above dataloader __init__() function

    # train_set = RoadSignSet(split='train', dataset_path=cfg.dataset_path, normalize=False)
    # test_set = RoadSignSet(split='test', dataset_path=cfg.dataset_path, normalize=False)
    # full_loader = torch.utils.data.ConcatDataset([train_set, test_set])
    # mean, std = calculate_mean_std(full_loader, recalculate=True)

    # Train data
    train_set = RoadSignSet(split='train', dataset_path=cfg.dataset_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Test data
    test_set = RoadSignSet(split='test', dataset_path=cfg.dataset_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Model
    model = cfg.model_class(num_classes=4)
    model.to(cfg.DEVICE)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Verifying random initialisation => ~25% accuracy
    best_accuracy = 0.
    accuracy = eval(data_loader=test_loader, model=model)
    print(col.BLUE, f"Warm up accuracy: {accuracy:.2f}", col.END)

    for epoch in range(cfg.EPOCHS):
        print(col.BLUE, f"[Epoch {epoch}]", col.END)
        train(data_loader=train_loader, model=model, optimizer=optimizer, criterion=criterion)

        accuracy = eval(data_loader=test_loader, model=model)
        if accuracy > best_accuracy:
            print(col.GREEN, f"New best accuracy: {accuracy:.2f}", col.END)
            best_accuracy = accuracy
            torch.save(model.state_dict(), cfg.model_path)
        else: 
            print(col.YELLOW, f"Accuracy: {accuracy:.2f}", col.END)

        if best_accuracy == 100.:
            # Won't get any better
            break

    # Training finished. Let's quantize the model

    # Workaround on Mac for quantized backend
    # https://github.com/pytorch/pytorch/issues/29327      
    if sys.platform == 'darwin':
      torch.backends.quantized.engine = 'qnnpack'
    
    # Quantize the model
    model_int8 = torch.quantization.quantize_dynamic(
                  model.to('cpu'),  # the original model
                  {torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d},  # a set of layers to dynamically quantize
                  dtype=torch.qint8)  # the target dtype for quantized weights
    
    # Quantised model not supported for mps backend
    cfg.DEVICE = torch.device('cpu')

    # Check quantised model accuracy
    accuracy = eval(data_loader=test_loader, model=model_int8)
    print(col.BLUE, f"Quantized Model Accuracy: {accuracy:.2f}")

    # Save Model
    torch.save(model_int8.state_dict(), cfg.model_path.replace(".zip", "_quant.zip"))

