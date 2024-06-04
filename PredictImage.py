import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import math
import random

Root_dir = "PlantDisease/plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/valid"
test_dir = "PlantDisease/plant_diseases_dataset/test"
Diseases_classes = os.listdir(train_dir)

train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

batch_size = 32
train_dataloader = DataLoader(
    train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(
    valid, batch_size, num_workers=2, pin_memory=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("Cuda is available")
        return torch.device("cuda")
    else:
        print("CPU is available")
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)  # Move images to the correct device
        labels = labels.to(device)  # Move labels to the correct device
        out = self(images)  # Generate predictions
        # Calculate loss, Compute the cross entropy loss between input logits and target.
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images = images.to(device)  # Move images to the correct device
        labels = labels.to(device)  # Move labels to the correct device
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        # Calculate accuracy, Compute the cross entropy loss between input logits and target.
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, x):  # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# Get the device
device = get_default_device()


def initialize_model():
    """Initialize the model and other necessary components"""
    global model, transform, test_images_dir, image_files, fig, train
    model = to_device(CNN_NeuralNet(3, len(train.classes)), device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    model = CNN_NeuralNet(3, 38)
    model.load_state_dict(torch.load(
        'plant_disease_model_5epochs.pth', map_location=device))
    model.eval()
    model.to(device)
    test_images_dir = 'PlantDisease/plant_diseases_dataset/test/test'
    image_files = [f for f in os.listdir(test_images_dir) if os.path.isfile(
        os.path.join(test_images_dir, f))]
    fig = plt.figure(figsize=(6, 4))


def accuracy(outputs, labels):
    """Calculate accuracy of predictions"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def calculate_accuracy(preds, labels):
    """Calculates the accuracy of the predictions"""
    return (preds == labels).float().mean().item()


def predict_image(img, model):
    """Converts image to array and return the predicted class with highest probability"""
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    prob, preds = torch.max(torch.nn.functional.softmax(yb, dim=1), dim=1)
    return train.classes[preds[0].item()], prob[0].item()


def show_prediction_with_confidence(image_path, model, transform):
    """Show prediction with confidence for a given image"""
    img = Image.open(image_path)
    transformed_img = transform(img)
    predicted_label, probability = predict_image(transformed_img, model)
    return predicted_label, probability


def PredictGivenImage(image_path):
    """Predict the given image"""
    initialize_model()
    return show_prediction_with_confidence(
        image_path, model, transform)
