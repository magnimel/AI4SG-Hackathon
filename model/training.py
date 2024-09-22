# %%
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import os

# %%
# Loading and normalizing the data.
# Define transformations for the training and test sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(root='./../archive/image-data', transform=transform)
print(len(train_dataset))

# %%
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = []
        for ext in ['png', 'jpg']:
            self.image_paths += glob.glob(os.path.join(root_dir, '*', f'*.{ext}'))
        class_set = set()
        for path in self.image_paths:
            class_set.add(os.path.dirname(path))
        self.class_lbl = { cls: i for i, cls in enumerate(sorted(list(class_set)))}
        print(self.class_lbl)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx], ImageReadMode.RGB).float()
        cls = os.path.basename(os.path.dirname(self.image_paths[idx]))
        label = self.class_lbl[cls]

        return self.transform(img), torch.tensor(label)


# %%
dataset = CustomDataset('./../archive/image-data', transform)
print(len(dataset))
splits = [0.5, 0.25, 0.25]

# %%
split_sizes = []
for sp in splits[:-1]:
    split_sizes.append(int(sp * len(dataset)))
split_sizes.append(len(dataset) - sum(split_sizes))


# %%
train_set, test_set, val_set = torch.utils.data.random_split(dataset, split_sizes)

# %%
dataloaders = {
    "train": DataLoader(train_set, batch_size=12, shuffle=True),
    "test": DataLoader(test_set, batch_size=12, shuffle=False),
    "val": DataLoader(val_set, batch_size=12, shuffle=False)
}

# %%
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = torch.nn.Linear(3072, 256)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
model = Network()

# %%
from torch.optim import Adam
 
    
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# %%
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 3)
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# %%
metrics = {
    'train': {
         'loss': [], 'accuracy': []
    },
    'val': {
         'loss': [], 'accuracy': []
    },
}

# %%
for epoch in range(30):
    ep_metrics = {
        'train': {'loss': 0, 'accuracy': 0, 'count': 0},
        'val': {'loss': 0, 'accuracy': 0, 'count': 0},
    }

    print(f'Epoch {epoch}')

    for phase in ['train', 'val']:
        print(f'-------- {phase} --------')
        for images, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                output = model(images.to(device))
                ohe_label = torch.nn.functional.one_hot(labels, num_classes=6)

                loss = criterion(output, ohe_label.float().to(device))

                correct_preds = labels.to(device) == torch.argmax(output, dim=1)
                accuracy = correct_preds.sum().float() / len(labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            ep_metrics[phase]['loss'] += loss.item()
            ep_metrics[phase]['accuracy'] += accuracy.item()
            ep_metrics[phase]['count'] += 1

        print(ep_metrics)

        if ep_metrics[phase]['count'] > 0:  # Check to avoid division by zero
            ep_loss = ep_metrics[phase]['loss'] / ep_metrics[phase]['count']
            ep_accuracy = ep_metrics[phase]['accuracy'] / ep_metrics[phase]['count']
        else:
            ep_loss, ep_accuracy = float('nan'), float('nan')

        print(f'Loss: {ep_loss}, Accuracy: {ep_accuracy}\n')

        metrics[phase]['loss'].append(ep_loss)
        metrics[phase]['accuracy'].append(ep_accuracy)


# %%



