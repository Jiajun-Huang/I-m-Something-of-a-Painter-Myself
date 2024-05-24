import argparse
import time
from collections import Counter
from pprint import pprint

from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data_loader import CustomDataset
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--en_wandb', type=bool, default=True)

    # sample interval
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)

    parser.add_argument('--nr_resnet', type=int, default=4)
    parser.add_argument('--nr_updownsample', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # init wandb
    if args.en_wandb:
        wandb.init(project='Cycle-GAN')
        wandb.config.update(args)
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    pprint(vars(args))

    # set seed
    torch.manual_seed(args.seed)

    # create model
    gan = Cycle_GAN()
    
    # create data loader
    data_loader = DataLoader(CustomDataset("data/photo_jpg", "data/monet_jpg", args.batch_size), batch_size=args.batch_size, shuffle=True)
    sample_data_loader = DataLoader(CustomDataset("data/photo_jpg", "data/monet_jpg", args.batch_size, augment=False), batch_size=args.batch_size, shuffle=False)
    # create optimizer
    optimizer = optim.Adam(gan.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan.to(device)
    
    # train model
    for epoch in range(args.epochs):
        gen_loss_avg = {}
        dis_loss_avg = {}
        for i, (real_A, real_B) in enumerate(tqdm(data_loader)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            gan.optimize(real_A, real_B)
            gen_loss, dis_loss = gan.get_losses()
            # sum up loss in the dictionary

            # logging iteration loss
            for k, v in gen_loss.items():
                if k in gen_loss_avg:
                    gen_loss_avg[k] += v
                else:
                    gen_loss_avg[k] = v
            for k, v in dis_loss.items():
                if k in dis_loss_avg:
                    dis_loss_avg[k] += v
                else:
                    dis_loss_avg[k] = v
        scheduler.step()

        gen_loss_avg = {k: v / i for k, v in gen_loss_avg.items()}
        dis_loss_avg = {k: v / i for k, v in dis_loss_avg.items()}
        gen_loss_avg_sum = sum(gen_loss_avg.values())
        dis_loss_avg_sum = sum(dis_loss_avg.values())
        if epoch % args.sample_interval == 0:
            # sample data
            real_A, real_B = next(iter(sample_data_loader))
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            fake_A, fake_B = gan.sample(real_A=real_A, real_B=real_B)
            # inverse process
            fake_A = CustomDataset.image_inverse_preprocess(fake_A)
            fake_B = CustomDataset.image_inverse_preprocess(fake_B)
            real_A = CustomDataset.image_inverse_preprocess(real_A)
            real_B = CustomDataset.image_inverse_preprocess(real_B)
            
            # save images and make grid
            save_image(make_grid(real_A, nrow=8), f'output/real_A_{epoch}.png')
            save_image(make_grid(real_B, nrow=8), f'output/real_B_{epoch}.png')
            save_image(make_grid(fake_A, nrow=8), f'output/fake_A_{epoch}.png')
            save_image(make_grid(fake_B, nrow=8), f'output/fake_B_{epoch}.png')
            
            if args.en_wandb:
                wandb.log({'real_A': [wandb.Image(real_A)], 'fake_B': [wandb.Image(fake_B)], 'real_B': [wandb.Image(real_B)], 'fake_A': [wandb.Image(fake_A)]})
            
        print(f'Epoch: {epoch}, Generator Loss: {gen_loss}, Discriminator Loss: {dis_loss}')
        if args.en_wandb:
            wandb.log({'train_gen_loss': gen_loss_avg_sum, 'train_dis_loss': dis_loss_avg_sum})
            wandb.log(gen_loss_avg)
            wandb.log(dis_loss_avg)
            wandb.log({'epoch': epoch})        
            # if i % args.save_interval == 0:
            #     torch.save(gan.state_dict(), 'model.pth')
            




            



    





