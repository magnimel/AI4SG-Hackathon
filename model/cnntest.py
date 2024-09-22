import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class GarbageClassifierCNN(nn.Module):
    def __init__(self):
        super(GarbageClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # 3 input channels (RGB), 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,6) # 6 because of the six different classes of categories
    
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


transform =transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



image_path = 'assets/testingImage/bottleTest.jpg'
image_p = 'assets/testingImage/metalTest.jpg'
image_pp ='assets/testingImage/paperTest.jpg'
image_ppp = 'assets/testingImage/cardboardTest.jpg'
image_pppp = 'assets/testingImage/bottleTest4.jpg'
image_ppppp = 'assets/testingImage/metalTest1.jpg'



image = Image.open(image_path)
image2 = Image.open(image_p)
image3 = Image.open(image_pp)
image4 = Image.open(image_ppp)
image5 = Image.open(image_pppp)
image6 = Image.open(image_ppppp)

image = transform(image).unsqueeze(0)
image2 = transform(image2).unsqueeze(0)
image3 = transform(image3).unsqueeze(0)
image4 = transform(image4).unsqueeze(0)
image5 = transform(image5).unsqueeze(0)
image6 =  transform(image6).unsqueeze(0)


images = [image, image2, image3, image4, image5,image6]
with torch.no_grad():
    for i in images:
        output = model(i)
        predicted_class = torch.argmax(output, dim=1)
        print(f'Predicted class: {classes[predicted_class.item()]}')

