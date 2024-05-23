import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import os

# Define the CNN_NeuralNet and ConvBlock classes
class CNN_NeuralNet(nn.Module):
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Define the path to your training data
train_dir = "PlantDisease/plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

Diseases_classes = os.listdir(train_dir)
print("Disease classes: ", Diseases_classes)
print("Total number of classes: ", len(Diseases_classes))

# Load the model
num_classes = len(Diseases_classes)  # Set this to the number of classes in your dataset
model = CNN_NeuralNet(3, num_classes)
model.load_state_dict(torch.load('plant_disease_model.pth'))
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict image
def predict_image(img, model):
    """Converts image to array and return the predicted class with highest probability"""
    # Convert to a batch of 1
    xb = img.unsqueeze(0).to(device)  # Move the tensor to the appropriate device
    # Get predictions from model
    yb = model(xb)
    print("yb is: ", yb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return Diseases_classes[preds[0].item()]

# Function to predict image with confidence score
def predict_image_with_confidence(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Move the tensor to the appropriate device
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item()

# Function to display prediction with confidence score below the image
def show_prediction_with_confidence(image_path, model, transform):
    img = Image.open(image_path)
    transformed_img = transform(img)
    
    predicted_label = predict_image(transformed_img, model)
    _, confidence = predict_image_with_confidence(image_path, model, transform)
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label}")
    plt.figtext(0.5, 0.01, f"Confidence: {confidence:.2f}", ha="center", fontsize=12, color="red")
    plt.show()

# Example usage
image_path = 'PlantDisease/plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_mosaic_virus/0a91f50b-1263-4b2c-a8c1-f2a6025b82f3___PSU_CG 2136.JPG'  # Replace with the path to your image

if os.path.exists(image_path):
    show_prediction_with_confidence(image_path, model, transform)
else:
    print(f"File does not exist: {image_path}")
