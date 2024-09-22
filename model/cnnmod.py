import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#dataset = datasets.ImageFolder(root='C:\Users\Owner\Downloads\archive\Garbage classification\Garbage classification', transform=transform)
dataset = datasets.ImageFolder(root=r'./../archive/image-data', transform=transform)

#dataset = datasets.ImageFolder(C:r'\Users\Owner\Downloads')
classes = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)


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
    # def __init__(self):
    #     super(GarbageClassifierCNN, self).__init__()

    #     self.linear1 = torch.nn.Linear(3072, 256)
    #     self.activation = torch.nn.ReLU()
    #     self.linear2 = torch.nn.Linear(256, 6)
    #     self.softmax = torch.nn.Softmax(dim=1)

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = self.linear1(x)
    #     x = self.activation(x)
    #     x = self.linear2(x)
    #     x = self.softmax(x)
    #     return x



model = GarbageClassifierCNN()
#initialize loss function and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
num_epochs =  15
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataLoader:
        optimizer.zero_grad()
        outputs= model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataLoader):.4f}')

#dummy_input = torch.randn(1, 3, 32, 32)

#model.eval()
#with torch.no_grad():
 #   output = model(dummy_input)
 #   print("Output shape: ", output.shape)
 #   print("Output: ", output)


#torch.argmax(output, dim=1)

torch.save(model.state_dict(), 'garbage_classifier.pth')