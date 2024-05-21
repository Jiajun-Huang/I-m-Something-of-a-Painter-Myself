import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
    parser.add_argument('--en_wandb', type=bool, default=True)

    # sample interval
    parser.add_argument('--sample_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=50)

    parser.add_argument('--nr_resnet', type=int, default=4)
    parser.add_argument('--nr_updownsample', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # init wandb
    if args.en_wandb:
        wandb.init(project='Cycle-GAN')
        wandb.config.update(args)


    # set seed
    torch.manual_seed(args.seed)

    # create model
    gan = Cycle_GAN()
    
    # create data loader
    data_loader = DataLoader(Dataset("data/photo_jpg", "data/monet_jpg", args.batch_size), batch_size=args.batch_size, shuffle=True)

    # create optimizer
    optimizer = optim.Adam(gan.parameters(), lr=args.lr)

    # check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan.to(device)
    
    # train model
    for epoch in range(args.epochs):
        for i, (real, fake) in enumerate(data_loader):
            real = real.to(device)
            fake = fake.to(device)
            gan.optimize(real, fake)

            # logging 
            gen_loss, dis_loss = gan.get_losses()
            if args.en_wandb:
                wandb.log({'train_gen_loss': gen_loss, 'train_dis_loss': dis_loss})

        if epoch % args.sample_interval == 0:
            fake_A, fake_B = gan.sample(real_A=real, real_B=fake)
            # inverse process
            fake_A = Dataset.image_inverse_preprocess(fake_A)
            fake_B = Dataset.image_inverse_preprocess(fake_B)
            # save images to output
            # plt.imshow(fake_A[0].permute(1, 2, 0))
            # plt.savefig(f'output/fake_A_{epoch}.png')
            # plt.imshow(fake_B[0].permute(1, 2, 0))
            # plt.savefig(f'output/fake_B_{epoch}.png')

            if args.en_wandb:
                wandb.log({'real_A': [wandb.Image(real)], 'fake_B': [wandb.Image(fake_B)], 'real_B': [wandb.Image(fake)], 'fake_A': [wandb.Image(fake_A)]})
            
        if args.en_wandb:
            wandb.log({'epoch': epoch})        
            # if i % args.save_interval == 0:
            #     torch.save(gan.state_dict(), 'model.pth')
            




            



    





