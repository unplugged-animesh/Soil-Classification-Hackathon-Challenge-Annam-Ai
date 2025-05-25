# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

df = pd.read_csv(train_csv_path)
df['label'] = 1 

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from sklearn.metrics import precision_recall_curve

def predict_with_tta(model, image, transforms_list):
    model.eval()
    predictions = []
    with torch.no_grad():
        for transform in transforms_list:
            augmented_image = transform(image)
            augmented_image = augmented_image.unsqueeze(0).to(device)
            output = model(augmented_image)
            predictions.append(output.item())
    return np.mean(predictions)

test_ids = pd.read_csv(test_ids_path)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    test_transform
]

def run_test_predictions(model, threshold=0.5):
    predictions = []
    for img_id in tqdm(test_ids['image_id'], desc="Predicting Test"):
        img_path = os.path.join(test_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        prob = predict_with_tta(model, image, tta_transforms)
        label = 1 if prob > threshold else 0
        predictions.append((img_id, label))
    return predictions