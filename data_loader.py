import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize
from tqdm import tqdm
import pandas as pd
import pdb


class Dataset(Dataset):
    def __init__(self, data_path, batch_size, sheffle=False):
        
        self.data_path = 
        self.batch_size = batch_size
        self.sheffle = sheffle
        self.data = os.listdir(self.data_path)

        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        image: (B, C, H, W)
        '''
        img_path = os.path.join(self.data_path, self.data[idx])
        image = read_image(img_path)
        
        # resize image to 286 x 286
        image = Resize((286, 286))(image)

        # random crop image
        i, j, h, w = RandomCrop.get_params(image, output_size=(256, 256))
        
       # random horizontal flip
        image = RandomHorizontalFlip()(image)

        # transform image to -1 to 1
        image = image.type(torch.float32) / 255

        # normalize image
        image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        return image
        

        
    



    
        