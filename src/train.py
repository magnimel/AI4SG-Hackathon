# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import zipfile
import os


# Configuration parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 9  # base on the class dataset we got
# was to unzip the dataset and use it, but we might not need it anymore

# ZIP_FILE_PATH = 'assets/dataZipedLocated/dataset.zip'  # Path to the zip file with the data sets
# UNZIP_PATH = 'data'  # Directory where the data will be extracted to, this way we wont overwhelm our computers
#
# # Unzip the dataset
# def unzip_dataset(zip_path, extract_path):
#     if not os.path.exists(extract_path):
#         os.makedirs(extract_path)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#     print(f"Dataset unzipped to: {extract_path}")
#
#
# # Run the unzip function
# unzip_dataset(ZIP_FILE_PATH, UNZIP_PATH)
#
# # Set the path to the unzipped data
# DATA_PATH = f'{UNZIP_PATH}'  # This will point to the root of your unzipped data directory
#
#
# # Set the path to the unzipped data
# DATA_PATH = f'{UNZIP_PATH}'  # This will point to the root of your unzipped data directory
