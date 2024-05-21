import torch
import torch.nn as nn
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
        self._gen_AB = Generator()
        self._gen_BA = Generator()
        self._dis_A = Discriminator()
        self._dis_B = Discriminator()

        # forward buffers
        self._fake_A = None
        self._fake_B = None
        self._rec_A = None
        self._rec_B = None
        self._real_A = None
        self._real_B = None
        self._pred_real_A = None
        self._pred_fake_A = None
        self._pred_real_B = None
        self._pred_fake_B = None

        # optimizers
        self._gen_opt = torch.optim.Adam(list(self._gen_AB.parameters()) + list(self._gen_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self._dis_opt = torch.optim.Adam(list(self._dis_A.parameters()) + list(self._dis_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

        # loss buffers
        self._gen_loss = None
        self._dis_loss = None

    def forward_G(self, real_A, real_B):
        # Forward pass through the generators
        fake_B = self._gen_AB(real_A)
        rec_A = self._gen_BA(fake_B) # reconstruct A 

        fake_A = self._gen_BA(real_B)
        rec_B = self._gen_AB(fake_A) # reconstruct B

        self._real_A = real_A
        self._real_B = real_B
        self._fake_A = fake_A
        self._fake_B = fake_B
        self._rec_A = rec_A
        self._rec_B = rec_B
    def forward_D(self, real_A, real_B):

        # Forward pass through the discriminators
        pred_real_A = self._dis_A(real_A)
        pred_fake_A = self._dis_A(self._fake_A.detach())
        pred_real_B = self._dis_B(real_B)
        pred_fake_B = self._dis_B(self._fake_B.detach())

        self._pred_real_A = pred_real_A
        self._pred_fake_A = pred_fake_A 
        self._pred_real_B = pred_real_B
        self._pred_fake_B = pred_fake_B
    
    def backward_G(self):
        # Calculate the generator loss
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        self._gen_opt.zero_grad()
        self._gen_AB.train()
        self._gen_BA.train()

        # cycle loss
        cycle_loss = l1_loss(self._real_A, self._rec_A) + l1_loss(self._real_B, self._rec_B)

        # generator loss
        dis_output_A = self._dis_A(self._fake_A).detach()
        dis_output_B = self._dis_B(self._fake_B).detach()
        gen_loss_A = mse_loss(dis_output_A, torch.ones_like(dis_output_A))
        gen_loss_B = mse_loss(dis_output_B, torch.ones_like(dis_output_B))

        # identity loss
        id_loss_A = l1_loss(self._real_A, self._gen_BA(self._real_A))
        id_loss_B = l1_loss(self._real_B, self._gen_AB(self._real_B))

        gen_loss = cycle_loss * 10 + id_loss_A + id_loss_B + gen_loss_A + gen_loss_B
        gen_loss.backward()
        self._gen_opt.step()
        return gen_loss

    def backward_D(self):
        # Calculate the discriminator loss MSE loss
        self._dis_opt.zero_grad()
        self._dis_A.train()
        self._dis_B.train()

        loss = nn.MSELoss()
        dis_loss_A = loss(self._pred_real_A, torch.ones_like(self._pred_real_A)) + loss(self._pred_fake_A, torch.zeros_like(self._pred_fake_A))
        dis_loss_B = loss(self._pred_real_B, torch.ones_like(self._pred_real_B)) + loss(self._pred_fake_B, torch.zeros_like(self._pred_fake_B))

        dis_loss = (dis_loss_A + dis_loss_B) / 2
        dis_loss.backward()
        self._dis_opt.step()
        return dis_loss

        
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

    def optimize(self, real_A, real_B):
        # optimize the generators
        self.set_requires_grad([self._dis_A, self._dis_B], False)
        self.forward_G(real_A, real_B)
        self._gen_loss = self.backward_G()

        # optimize the discriminators
        self.set_requires_grad([self._dis_A, self._dis_B], True)
        self.forward_D(real_A, real_B)
        self._dis_loss = self.backward_D()
    
    def sample(self, real_A, real_B):
        fake_B = self._gen_AB(real_A)
        fake_A = self._gen_BA(real_B)
        return fake_A, fake_B

    def get_losses(self):
        return self._gen_loss, self._dis_loss


    
        



    
    