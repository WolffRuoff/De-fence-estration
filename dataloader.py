import os
import random
import cv2
from cv2 import inpaint
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def Get_Image_Paths():
    baseball_field = [f'Images/baseball_field/{file}' for file in os.listdir('Images/baseball_field') if '.jpg' in file]
    canyon = [f'Images/canyon/{file}' for file in os.listdir('Images/canyon') if '.jpg' in file]
    desert_sand = [f'Images/desert_sand/{file}' for file in os.listdir('Images/desert_sand') if '.jpg' in file]
    field_wild = [f'Images/field_wild/{file}' for file in os.listdir('Images/field_wild') if '.jpg' in file]
    formal_garden = [f'Images/formal_garden/{file}' for file in os.listdir('Images/formal_garden') if '.jpg' in file]
    ruin = [f'Images/ruin/{file}' for file in os.listdir('Images/ruin') if '.jpg' in file]
    image_paths = baseball_field + canyon + desert_sand + field_wild + formal_garden + ruin
    np.random.shuffle(image_paths)
    return image_paths

def Overlay_Fence(img, fence):
    combined_img = Image.new('RGBA', img.size)
    combined_img = Image.alpha_composite(combined_img, img)
    return Image.alpha_composite(combined_img, fence)

def Create_Mask(fence):
    fence_np = np.array(fence)
    #print(fence_np.shape)
    ret, mask = cv2.threshold(fence_np[:, :, 3], 50, 255, cv2.THRESH_BINARY)
    return mask

def dilate_image(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((3,3), np.uint8)
    dil_mask = cv2.dilate(mask, kernel, iterations=1)
    return dil_mask
class FenceDataset(Dataset):
    def __init__(self, image_paths, fence_paths, img_transforms=None, fence_transforms=None, combined_transforms=None, inpainting=False):
        self.images_paths = image_paths
        self.fence_paths = fence_paths
        self.img_transforms = img_transforms
        self.fence_transforms = fence_transforms
        self.combined_transforms = combined_transforms
        self.inpainting = inpainting
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index]).convert("RGBA")
        fence = Image.open(random.choice(self.fence_paths))

        if self.img_transforms:
            img = self.img_transforms(img)
        if self.fence_transforms:
            fence = self.fence_transforms(fence)
        mask = Create_Mask(fence)
        combined_img = Overlay_Fence(img, fence)
        
        # Convert to tensors
        totensor = transforms.ToTensor()
        #if self.inpainting:
        #    img = totensor(img.convert("RGB"))

        combined_img = totensor(combined_img.convert("RGB"))
        
        if self.combined_transforms and self.inpainting:
            img = totensor(img.convert("RGB"))
            img = self.combined_transforms(img)
            combined_img = self.combined_transforms(combined_img)
            mask = dilate_image(mask)
            mask = totensor(mask)
            return combined_img, img, mask
        elif self.combined_transforms:
            combined_img = self.combined_transforms(combined_img)
            mask = totensor(mask)
            return combined_img, mask
        elif self.inpainting:
            img = totensor(img.convert("RGB"))
            mask = dilate_image(mask)
            mask = totensor(mask)
            return combined_img, img, mask
        else:
            return combined_img, mask
        

def Get_DataLoaders(batch_size, num_workers, inpainting=False):
    fence_paths = [f'Fences/{file}' for file in os.listdir('Fences/') if '.png' in file]
    image_paths = Get_Image_Paths()

    bounds = [int(len(image_paths)*0.8), int(len(image_paths)*0.9)]
    fence_bounds = [int(len(fence_paths)*0.6), int(len(fence_paths)*0.9)]
    
    image_train, fence_train = image_paths[:bounds[0]], fence_paths[:fence_bounds[0]]
    image_val, fence_val = image_paths[bounds[0]:bounds[1]], fence_paths[fence_bounds[0]:fence_bounds[1]]
    image_test, fence_test = image_paths[bounds[1]:], fence_paths[fence_bounds[1]:]
    print(f"Image train={len(image_train)}, Fence train={len(fence_train)}")
    print(f"Image val={len(image_val)}, Fence val={len(fence_val)}")
    print(f"Image test={len(image_test)}, Fence test={len(fence_test)}")

    # Creating the transforms
    resize_crop = transforms.RandomResizedCrop((256,256), scale=(0.05, 1))
    resize = transforms.Resize((256,256))
    #crop = transforms.RandomCrop((256, 256))
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    jitter = transforms.ColorJitter(brightness=0.3, hue=0.1)
    blur = transforms.GaussianBlur(7, (0.1,10))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img_transforms = transforms.Compose((resize, hflip, jitter))
    fence_transforms = transforms.Compose((resize_crop, hflip, blur))
    #combined_transforms = transforms.Compose((crop, jitter))
    
    if inpainting:
        train_dataset = FenceDataset(image_train, fence_train, img_transforms, fence_transforms, normalize, inpainting)
        val_dataset = FenceDataset(image_val, fence_val, fence_transforms=fence_transforms, combined_transforms=normalize, inpainting=True)
        test_dataset = FenceDataset(image_test, fence_test, fence_transforms=fence_transforms, combined_transforms=normalize, inpainting=True)

        loader_train = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        loader_val = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
        loader_test = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)

        # View a few samples
        figure = plt.figure(figsize=(30, 10))
        cols, rows = 5, 3
        img_batch, label_batch, mask_batch, = next(iter(loader_train))
        for i in range(1, cols+1):
            input, label, mask = img_batch[i], label_batch[i], mask_batch[i]
            figure.add_subplot(rows, cols, i)
            plt.title(f"Image {i}")
            plt.axis("off")
            plt.imshow(unnormalize(input).permute(1, 2, 0))
            j = cols + i
            figure.add_subplot(rows, cols, j)
            plt.title(f"Label {i}")
            plt.axis("off")
            plt.imshow(unnormalize(label).permute(1, 2, 0))
            k = (cols * 2) + i
            figure.add_subplot(rows, cols, k)
            plt.title(f"mask {i}")
            plt.axis("off")
            plt.imshow(mask.permute(1,2,0))
        plt.show()

        return loader_train, loader_val, loader_test
    else:
        train_dataset = FenceDataset(image_train, fence_train, img_transforms, fence_transforms, normalize)
        val_dataset = FenceDataset(image_val, fence_val, fence_transforms=fence_transforms, combined_transforms=normalize)
        test_dataset = FenceDataset(image_test, fence_test, fence_transforms=fence_transforms, combined_transforms=normalize)

        loader_train = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        loader_val = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
        loader_test = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)

        # View a few samples
        figure = plt.figure(figsize=(30, 10))
        cols, rows = 5, 2
        img_batch, mask_batch, = next(iter(loader_train))
        for i in range(1, cols+1):
            input, mask = img_batch[i], mask_batch[i]
            figure.add_subplot(rows, cols, i)
            plt.title(f"Image {i}")
            plt.axis("off")
            plt.imshow(unnormalize(input).permute(1, 2, 0))
            j = cols + i
            figure.add_subplot(rows, cols, j)
            plt.title(f"Mask {i}")
            plt.axis("off")
            plt.imshow(mask.permute(1, 2, 0))
        plt.show()

        return loader_train, loader_val, loader_test
