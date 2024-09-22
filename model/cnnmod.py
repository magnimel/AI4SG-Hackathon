import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=r'archive/Garbage classificationMerged/Garbage classification', transform=transform)

# Define dataset split ratio (e.g., 80% for training, 20% for testing)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for both training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define the 6 classes for garbage categories
classes = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')

# Define the CNN model
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
        x = x.view(-1, 32 * 8 * 8)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = GarbageClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()  # Set model to training mode
    
    for images, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        
        running_loss += loss.item()  # Accumulate loss
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        total += labels.size(0)  # Total number of labels
        correct += (predicted == labels).sum().item()  # Correct predictions
    
    epoch_loss = running_loss / len(train_loader)  # Average loss per epoch
    accuracy = 100 * correct / total  # Accuracy percentage
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'garbage_classifier.pth')

# Testing loop
model.eval()  # Set model to evaluation mode (no gradient calculation)
test_correct = 0
test_total = 0

with torch.no_grad():  # Disable gradient calculation for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Output the predicted class names for each image in the batch
        #uncomment this part is you want to see the test predictions
        # for i in range(images.size(0)):
        #     print(f'Predicted: {classes[predicted[i]]}, Actual: {classes[labels[i]]}')

test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')
