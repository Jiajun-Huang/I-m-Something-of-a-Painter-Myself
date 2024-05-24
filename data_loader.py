import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
class CustomDataset(Dataset):
    def __init__(self, real_image_path, fake_image_path, batch_size, shuffle=False, augment=True):
        '''
        real_image_path: str
        fake_image_path: str
        '''
        self.real_image_path = real_image_path
        self.fake_image_path = fake_image_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.real_images = os.listdir(real_image_path)
        self.fake_images = os.listdir(fake_image_path)
        self.augment = augment
        if self.shuffle:
            random.shuffle(self.real_images)
            random.shuffle(self.fake_images)

    def image_preprocess(self, image):
        '''
        image: (C, H, W)
        '''
        H, W = image.shape[1:]

        if self.augment:
        # resize image to 1.12
            image = Resize((int(H * 1.12), int(W * 1.12)))(image)
            # random crop image
            i, j, h, w = RandomCrop.get_params(image, output_size=(H, W))
            image = image[:, i:i+h, j:j+w]
            # random horizontal flip
            image = RandomHorizontalFlip()(image)

        # transform image to [0, 1]
        image = image / 255.0
        # normalize image
        image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        return image

    @staticmethod
    def image_inverse_preprocess(image):
        '''
        image: (B, C, H, W)
        '''
        device = image.device   
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)

        # Denormalize the image
        image = image * std[None, :, None, None] + mean[None, :, None, None]
        return image

    def __len__(self):
        return max(len(self.real_images), len(self.fake_images))
    
    def __getitem__(self, idx):
        '''
        image: (C, H, W)
        '''
        idx_real_A = idx % len(self.real_images)
        idx_real_B = idx % len(self.fake_images)
        real_A_img_path = os.path.join(self.real_image_path, self.real_images[idx_real_A])
        real_B_img_path = os.path.join(self.fake_image_path, self.fake_images[idx_real_B])
        real_A_image = read_image(real_A_img_path)
        real_B_image = read_image(real_B_img_path)
        real_A_image = self.image_preprocess(real_A_image)
        real_B_image = self.image_preprocess(real_B_image)

        return real_A_image, real_B_image

if __name__ == '__main__':
    data_loader = DataLoader(CustomDataset("data/photo_jpg", "data/monet_jpg", 1), batch_size=1, shuffle=False)
    
    for i, (real, fake) in enumerate(data_loader):
        real = CustomDataset.image_inverse_preprocess(real)
        fake = CustomDataset.image_inverse_preprocess(fake)

        save_image(real, f"real_A_image_{i}.png")
        save_image(fake, f"real_B_image_{i}.png")

        plt.imshow(real[0].permute(1, 2, 0))
