import torch.nn as nn 
import torch, net
from model.modules import Generator, Discriminator 

class Cycle_GAN(nn.Module):
    '''
    Cycle GAN model
        Generator
        real A -> fake B -> rec A
        real B -> fake A -> rec B
        
        Optimizing the generator:
            L1 loss between real A and rec A
            L1 loss between real B and rec B
            Want discriminator to predict fake A and fake B as real
            
        Optimizing the discriminator:
            Discriminator should predict real A and real B as real and fake A and fake B as fake

    
    
    '''
    def __init__(self):
        super(Cycle_GAN, self).__init__()
        # Define the generator and discriminator
        self.gen_AB = Generator()
        self.gen_BA = Generator()
        self.dis_A = Discriminator()
        self.dis_B = Discriminator()

        # state variables
        self.fake_A = None
        self.fake_B = None
        self.rec_A = None
        self.rec_B = None
        self.real_A = None
        self.real_B = None
        self.pred_real_A = None
        self.pred_fake_A = None
        self.pred_real_B = None
        self.pred_fake_B = None

        # optimizers
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))

        self.dis_opt = torch.optim.Adam(list(self.dis_A.parameters()) + list(self.dis_B.parameters()), lr=0.0002, betas=(0.5, 0.999))


    def forward(self, real_A, real_B):
        # Forward pass through the generators
        fake_B = self.gen_AB(real_A)
        rec_A = self.gen_BA(fake_B) # reconstruct A 

        fake_A = self.gen_BA(real_B)
        rec_B = self.gen_AB(fake_A) # reconstruct B

        self.real_A = real_A
        self.real_B = real_B
        self.fake_A = fake_A
        self.fake_B = fake_B
        self.rec_A = rec_A
        self.rec_B = rec_B

        # Forward pass through the discriminators
        pred_real_A = self.dis_A(real_A)
        pred_fake_A = self.dis_A(fake_A)
        pred_real_B = self.dis_B(real_B)
        pred_fake_B = self.dis_B(fake_B)

        self.pred_real_A = pred_real_A
        self.pred_fake_A = pred_fake_A 
        self.pred_real_B = pred_real_B
        self.pred_fake_B = pred_fake_B
    
    def backward_G(self):
        # Calculate the generator loss
        cycle_loss = nn.L1Loss(self.real_A, self.rec_A) + nn.L1Loss(self.real_B, self.rec_B)
        gen_loss_A = nn.MSELoss(self.pred_fake_A, torch.ones_like(self.pred_fake_A))
        gen_loss_B = nn.MSELoss(self.pred_fake_B, torch.ones_like(self.pred_fake_B))

        # identity loss
        id_loss_A = nn.L1Loss(self.real_A, self.gen_BA(self.real_A))
        id_loss_B = nn.L1Loss(self.real_B, self.gen_AB(self.real_B))

        gen_loss = gen_loss_A + gen_loss_B + cycle_loss * 10 + id_loss_A + id_loss_B
        gen_loss.backward()
        self.gen_opt.step()

    def backward_D(self):
        # Calculate the discriminator loss MSE loss
        dis_loss_A = nn.MSELoss(self.pred_real_A, torch.ones_like(self.pred_real_A)) + nn.MSELoss(self.pred_fake_A, torch.zeros_like(self.pred_fake_A))
        dis_loss_B = nn.MSELoss(self.pred_real_B, torch.ones_like(self.pred_real_B)) + nn.MSELoss(self.pred_fake_B, torch.zeros_like(self.pred_fake_B))

        dis_loss = dis_loss_A + dis_loss_B
        dis_loss.backward()
        self.dis_opt.step()

        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize(self):
        # optimize the generators
        self.gen_opt.zero_grad()
        self.set_requires_grad([self.dis_A, self.dis_B], False)
        self.set_requires_grad([self.gen_AB, self.gen_BA], True)
        self.backward_G()

        # optimize the discriminators
        self.dis_opt.zero_grad()
        self.set_requires_grad([self.dis_A, self.dis_B], True)
        self.set_requires_grad([self.gen_AB, self.gen_BA], False)
        self.backward_D()



        



    
    