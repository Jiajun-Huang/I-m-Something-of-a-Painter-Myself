import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, C, dim, dim)
        '''
        x = x + self.model(x)

        return x

class Generator(nn.module):

    def __init__(self, dim=100, in_out_channel = 3, num_residual_blocks=6, num_up_downsampling=2):
        super(Generator, self).__init__()
        # use covolutional layers
        self.model = []

        # down sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2** i
            c_out = in_out_channel * 2 ** (i+1)
            self.model += [
                nn.Conv2d(c_in, c_out, 7, 1, 3),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ]

        # residual blocks
        for i in range(num_residual_blocks):
            self.model += [ResnetBlock(dim)]
        
        # up sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2 ** (num_up_downsampling - i)
            c_out = in_out_channel * 2 ** (num_up_downsampling - i - 1)
            self.model += [
                nn.ConvTranspose2d(c_in, c_out, 3, 2, 1, 1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ]
        
        self.model = nn.Sequential(*self.model)
        
        

    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, C, dim, dim)
        '''
        x = self.model(x)

        return x
    

class Discriminator(nn.module):

    def __init__(self, dim=100, num_upsampling=2):
        super(Discriminator, self).__init__()
        # up sampling
        self.model = []

        for i in range(num_upsampling):
            c_in = 3
            c_out = 3 * 2 ** (i+1)
            self.model += [
                nn.Conv2d(c_in, c_out, 3, 2, 1),
                nn.InstanceNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.model += [
            nn.Conv2d(c_out, 1, 3, 1, 1)
        ]

        self.model = nn.Sequential(*self.model)


    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, 1, dim, dim)
        '''

        x = self.model(x)

        return x