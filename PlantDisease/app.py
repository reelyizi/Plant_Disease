import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import colorama
from colorama import Fore, Style
import matplotlib
import time
import random

def main():
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

    def save_model(model, path):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def show_image(image, label):
        print("Label :" + train.classes[label] + "(" + str(label) + ")")
        plt.imshow(image.permute(1, 2, 0))

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            print("Cuda is available")
            return torch.device("cuda")
        else:
            print("CPU is available")
            return torch.device("cpu")

    matplotlib.use('Agg')

    # for moving data to device (CPU or GPU)
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    # for loading in the device (GPU if available else CPU)
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dataloader, device):
            self.dataloader = dataloader
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dataloader:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dataloader)

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

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

    # convolution block with BatchNormalization
    def ConvBlock(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    device = get_default_device()
    print(f"Using device: {device}")  # Debug statement

    # resnet architecture
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

    # defining the model and moving it to the GPU
    # 3 is number of channels RGB, len(train.classes()) is number of diseases.
    model = to_device(CNN_NeuralNet(3, len(train.classes)), device)
    print(f"Model is on device: {next(model.parameters()).device}")  # Debug statement

    # for training
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                     grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []  # For collecting the results
        val_acc_history = []  # For collecting validation accuracy
        val_loss_history = []  # For collecting validation loss

        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        # scheduler for one cycle learning rate
        # Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        # The 1cycle policy anneals the learning rate from an initial learning rate to some
        # maximum learning rate and then from that maximum learning rate to some minimum learning rate
        # much lower than the initial learning rate.
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                                    epochs=epochs, steps_per_epoch=len(train_loader))

        start_time = time.time() 
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # gradient clipping
                # Clip the gradients of an iterable of parameters at specified value.
                # All from pytorch documentation.
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # recording and updating learning rates
                lrs.append(get_lr(optimizer))
                sched.step()
                # validation

            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
            
            # Collect validation accuracy and loss
            val_acc_history.append(result['val_acc'])
            val_loss_history.append(result['val_loss'])

        end_time = time.time()  # End the timer
        total_time = end_time - start_time
        print(f"Training completed in: {total_time // 60:.0f} minutes {total_time % 60:.0f} seconds")
        return history

    num_epoch = 5
    lr_rate = 0.01
    grad_clip = 0.15
    weight_decay = 1e-4
    optims = torch.optim.Adam
    #history = fit_OneCycle(epochs=1, max_lr=0.01, model=model, train_loader=train_dataloader, val_loader=valid_dataloader, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=optims)
    history = fit_OneCycle(epochs=1, max_lr=lr_rate, model=model, train_loader=train_dataloader, val_loader=valid_dataloader, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=optims)

    #print(history)
    
    save_model(model, "plant_disease_model_1epochs.pth")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

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
        plt.title(f"Predicted: {predicted_label}")
        print("predic: ", predicted_label, " actual: ", randomImage)
        return i + 1

    fig = plt.figure(figsize=(8, 6))
    
    test_images_dir = 'PlantDisease/plant_diseases_dataset/test/test'
    image_files = [f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))]

    fig = plt.figure(figsize=(12, 12))
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

    val_acc = []
    val_loss = []
    train_loss = []

    for i in history:
        val_acc.append(i['val_acc'])
        val_loss.append(i['val_loss'])
        train_loss.append(i['train_loss'])

    plt.plot(val_acc, '-x', label="Validation Accuracy")
    plt.plot(val_loss, '-o', label="Validation Loss")
    plt.plot(train_loss, '-*', label="Train Loss")
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    
if __name__ == "__main__":
    main()