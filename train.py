import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import Dataset
from model.cyclegan import Cycle_GAN
from model.modules import Discriminator, Generator


def train(model, data_loader, optimizer, args):
    '''
    model: Cycle_GAN model
    data_loader: DataLoader
    optimizer: optimizer
    args: arguments
    '''
    # set model to train mode
    model.train()
    
    # set loss function
    criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        for i, data in enumerate(data_loader):
            pass
           


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--en_wandb', type=bool, default=False)

    # sample interval
    parser.add_argument('--sample_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=50)

    parser.add_argument('--nr_resnet', type=int, default=4)
    parser.add_argument('--nr_updownsample', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)

    # create model
    gan = Cycle_GAN()

    # create data loader
    data_loader = DataLoader(Dataset("data/photo_jpg", "data/monet_jpg", args.batch_size), batch_size=args.batch_size, shuffle=True)

    # create optimizer
    optimizer = optim.Adam(gan.parameters(), lr=args.lr)

    
    # train model
    for epoch in range(args.epochs):
        for i, (real, fake) in enumerate(data_loader):
            gan.optimize(real, fake)
            if i % args.sample_interval == 0:
                # gan.sample()
                pass
            if i % args.save_interval == 0:
                torch.save(gan.state_dict(), 'model.pth')
            print(f'Epoch: {epoch}, Iter: {i}')



            



    





