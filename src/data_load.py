from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import v2
from pathlib import Path
import torch
import os 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split



cwd = Path.cwd()
parent = cwd.parent

def data_load():
   train_transforms = v2.Compose([
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Resize((224,224)),
      v2.Grayscale(),
      v2.ColorJitter(brightness=0.1, contrast=0.1),
      v2.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1), antialias=True)])

   test_transforms = v2.Compose([
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Resize((224,224)),
      v2.Grayscale()])

   train_dir  = os.path.join(parent, "data", "Train")
   test_dir  = os.path.join(parent, "data", "Test")

   dummy_dataset = ImageFolder(train_dir)
   targets = dummy_dataset.targets
   train_idx, val_idx = train_test_split(np.arange(len(dummy_dataset)), test_size= 0.2, random_state=1, shuffle=True, stratify=targets)
   

   dataset_train = Subset(ImageFolder(train_dir, transform = train_transforms), train_idx)
   dataset_val= Subset(ImageFolder(train_dir, transform = test_transforms), val_idx)


   dataset_test = ImageFolder(test_dir, transform = test_transforms)

   train_loader = DataLoader(dataset_train, shuffle = True, batch_size=32, num_workers = 1, pin_memory = True, persistent_workers=True)
   val_loader = DataLoader(dataset_val, shuffle = False, batch_size=32, num_workers = 1, pin_memory = True, persistent_workers=True)
   test_loader = DataLoader(dataset_test, shuffle = False, batch_size=32,  num_workers = 1, pin_memory = True, persistent_workers=True)
   return train_loader, val_loader, test_loader

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = data_load()
    images, label = next(iter(val_loader)) 
    
    imshow(torchvision.utils.make_grid(images))
    
    
    