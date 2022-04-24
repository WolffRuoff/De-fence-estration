import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def Overlay_Fence(img, fence):
    combined_img = Image.new('RGBA', img.size)
    combined_img = Image.alpha_composite(combined_img, img)
    return Image.alpha_composite(combined_img, fence)

class FenceDataset(Dataset):
    def __init__(self, image_paths, fence_paths, img_transforms=None, fence_transforms=None, combined_transforms=None):
        self.images_paths = image_paths
        self.fence_paths = fence_paths
        self.img_transforms = img_transforms
        self.fence_transforms = fence_transforms
        self.combined_transforms = combined_transforms
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img = Image.open(self.images_paths(index)).convert("RGBA")
        fence = Image.open(random.choice(self.fence_paths))

        if self.img_transforms:
            img = self.img_transforms(img)
        if self.fence_transforms:
            fence = self.fence_transforms(fence)
        
        combined_img = Overlay_Fence(img, fence)
        
        if self.combined_transforms:
            combined_img = self.combined_transforms(combined_img)

        return combined_img, img

def Get_DataLoaders(batch_size, num_workers):
    fence_paths = [f'Fences/{file}' for file in os.listdir('Fences/') if '.png' in file]
    image_paths = [f'Images/{file}' for file in os.listdir('Images/') if '.png' in file or '.jpeg' in file or '.jpg' in file]
    # TODO Split paths for train and test

    # Creating the transforms
    crop = transforms.RandomCrop((480, 640))
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    jitter = transforms.ColorJitter(brightness=0.4, hue=0.2)
    blur = transforms.GaussianBlur(9, (0.1,15))

    img_transforms = transforms.Compose((hflip, crop))
    fence_transforms = transforms.Compose((crop, hflip, blur))
    combined_transforms = transforms.Compose((crop, jitter))

    train_dataset = FenceDataset(image_paths, fence_paths, img_transforms, fence_transforms, combined_transforms)
    loader_train = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    # View a few samples
    figure = plt.figure(figsize=(30, 10))
    cols, rows = 5, 2
    img_batch, mask_batch = next(iter(loader_train))
    for i in range(1, int((cols * rows)/2 + 1)):
        input, label = img_batch[i], mask_batch[i]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(input)
        j = cols + i
        figure.add_subplot(rows, cols, j)
        plt.title(f"Label {i}")
        plt.axis("off")
        plt.imshow(label)
    plt.show()

    return loader_train
