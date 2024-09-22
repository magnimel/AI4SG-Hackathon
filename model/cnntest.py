import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import warnings 

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress the warning

class GarbageClassifierCNN(nn.Module):
    def __init__(self):
        super(GarbageClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 input channels (RGB), 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # 6 different classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = GarbageClassifierCNN()
model.load_state_dict(torch.load('garbage_classifier.pth'))
model.eval()
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prompt for user input and split by whitespace
image_inputs = input("Enter the image names (without extension, separated by spaces): ").split()

for image_input in image_inputs:
    image_path = f'assets/testingImage/{image_input}.jpg'

    if not os.path.isfile(image_path):
        print(f"File {image_path} not found.")
    else:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1)
            print(f'Predicted class for {image_input}: {classes[predicted_class.item()]}')
