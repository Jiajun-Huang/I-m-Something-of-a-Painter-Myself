import torch.nn as nn
import torch

class ResnetBlock(nn.Module):
    def __init__(self, channel=3):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.InstanceNorm2d(channel)
        )

    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, C, dim, dim)
        '''
        x = x + self.model(x)

        return x

class Generator(nn.Module):

    def __init__(self, dim=256, in_out_channel = 3, num_residual_blocks=3):
        super(Generator, self).__init__()
        # use covolutional layers
        self.model = []
        num_up_downsampling = 2
        # down sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2** i
            c_out = in_out_channel * 2 ** (i+1)
            self.model += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), # [(W−K+2P)/S]+1  256 - 3 + 1 = 254
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ]

        # residual blocks
        for i in range(num_residual_blocks):
            channel = c_out
            self.model += [ResnetBlock(channel)]
        
        # up sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2 ** (num_up_downsampling - i)
            c_out = in_out_channel * 2 ** (num_up_downsampling - i - 1)
            self.model += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, output_padding=1, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ]
        self.model += [nn.ReflectionPad2d(3)]
        self.model += [nn.Conv2d(3, 3, kernel_size=13, padding=0)]

        self.model = nn.Sequential(*self.model)
        
        self.readout = nn.Parameter(torch.randn(dim, dim)) # solve chessboard effect
        
        

    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, C, dim, dim)
        '''
        B, C, H, W = x.shape
        x = self.model(x)
        x = x + self.readout[None, None, :, :]
        return x
    

class Discriminator(nn.Module):

    def __init__(self, num_upsampling=2):
        super(Discriminator, self).__init__()
        # up sampling
        self.model = []

        for i in range(num_upsampling):
            c_in = 3 * 2 ** i
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