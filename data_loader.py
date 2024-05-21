import os
import pdb

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize)
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, real_image_path,fake_image_path, batch_size, sheffle=False):
        '''
        data_path: str
        '''
        self.real_image_path = real_image_path
        self.fake_image_path = fake_image_path
        self.batch_size = batch_size
        self.sheffle = sheffle
        self.real_images = os.listdir(real_image_path)
        self.fake_images = os.listdir(fake_image_path)

    def image_preprocess(self, image):
        '''
        image: (B, C, H, W)
        '''
        # resize image to 286 x 286
        image = Resize((286, 286))(image)

        # random crop image
        i, j, h, w = RandomCrop.get_params(image, output_size=(256, 256))
        image = image[:, i:i+h, j:j+w]

        # random horizontal flip
        image = RandomHorizontalFlip()(image)

        # transform image to -1 to 1
        image = image.type(torch.float32) / 255

        # normalize image
        image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        return image

    def image_inverse_preprocess(image):
        '''
        image: (B, C, H, W)
        '''
        device = image.device   
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)

        # Denormalize the image
        image = image * std[None, :, None, None] + mean[None, :, None, None]

        # Transform image back to [0, 255]
        image = image * 255.0
        image = image.type(torch.uint8)
        return image

    def __len__(self):
        return len(max(self.real_images, self.fake_images, key=len))
    
    def __getitem__(self, idx):
        '''
        image: (B, C, H, W)
        '''
        idx_real = idx % len(self.real_images)
        idx_fake = idx % len(self.fake_images)
        real_img_path = os.path.join(self.real_image_path, self.real_images[idx_real])
        fake_img_path = os.path.join(self.fake_image_path, self.fake_images[idx_fake])
        real_image = read_image(real_img_path)
        fake_image = read_image(fake_img_path)
        
        real_image = self.image_preprocess(real_image)
        fake_image = self.image_preprocess(fake_image)

        return real_image, fake_image
        

if __name__ == '__main__':
    data_loader = DataLoader(Dataset("data/photo_jpg", "data/monet_jpg", 1), batch_size=1)
    
    for i, (real, fake) in enumerate(data_loader):
        real_image = Dataset.image_inverse_preprocess(real)
        fake_image = Dataset.image_inverse_preprocess(fake)

        plt.imshow(real_image[0].permute(1, 2, 0))
        plt.imshow(fake_image[0].permute(1, 2, 0))
        plt.show()


        
    



    
        