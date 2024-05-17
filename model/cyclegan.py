import torch.nn as nn

class Cycle_GAN(nn.Module):
    
    def __init__(self):
        super(Cycle_GAN, self).__init__()
        # Define the generator and discriminator
        self.gen_AB = Generator()
        self.gen_BA = Generator()
        self.dis_A = Discriminator()
        self.dis_B = Discriminator()
    
    def forward(self, x):
        # Forward pass through the generators
        x_AB = self.gen_AB(x)
        x_BA = self.gen_BA(x)
        # Forward pass through the discriminators
        d_A = self.dis_A(x)
        d_B = self.dis_B(x)
        return x_AB, x_BA, d_A, d_B
    