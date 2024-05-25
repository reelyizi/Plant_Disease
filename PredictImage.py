import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from colorama import Fore, Style
import matplotlib
import random

Root_dir = "PlantDisease/plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/valid"
test_dir = "PlantDisease/plant_diseases_dataset/test"
Diseases_classes = os.listdir(train_dir)

train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

batch_size = 32
train_dataloader = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)

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
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            images = images.to(device)  # Move images to the correct device
            labels = labels.to(device)  # Move labels to the correct device
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
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
            #self.conv5 = ConvBlock(256, 256, pool=True)
            #self.conv6 = ConvBlock(256, 512, pool=True)
            #self.conv7 = ConvBlock(512, 512, pool=True)

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
            #out = self.conv5(out)
            #out = self.conv6(out)
            #out = self.conv7(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out

device = get_default_device()
model = to_device(CNN_NeuralNet(3, len(train.classes)), device)

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def predict_image(img, model):
        """Converts image to array and return the predicted class
           with highest probability"""
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        
        return train.classes[preds[0].item()]

def show_prediction_with_confidence(image_path, model, transform, i, randomImage):
        img = Image.open(image_path)
        transformed_img = transform(img)
        
        predicted_label = predict_image(transformed_img, model)
        
        fig.add_subplot(3, 2, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {predicted_label} /N actual: {randomImage}")
        print("predic: ", predicted_label, " actual: ", randomImage)
        return i + 1


num_classes = len(Diseases_classes)  # Set this to the number of classes in your dataset
model = CNN_NeuralNet(3, num_classes)
model.load_state_dict(torch.load('plant_disease_model_1epochs.pth'))
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

fig = plt.figure(figsize=(6, 4))
    
test_images_dir = 'PlantDisease/plant_diseases_dataset/test/test'
image_files = [f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))]

i = 1  # Start with 1 because subplot indices are 1-based

    # Number of images to display
num_images = 6

for _ in range(num_images):
        # Randomly select an image
    random_image = random.choice(image_files)
    image_path = os.path.join(test_images_dir, random_image)
        
        # Check if the image path exists
    if os.path.exists(image_path):
        i = show_prediction_with_confidence(image_path, model, transform, i, random_image)

plt.show()