import torch
import torch.nn as nn


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

    def __init__(self, in_out_channel = 3, num_residual_blocks=9, device='cuda'):
        super(Generator, self).__init__()
        # use covolutional layers
        # self.model = []
        self.downsamp = []
        self.upsamp = []
        self.resnet = []
        num_up_downsampling = 2
        # down sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2** i
            c_out = in_out_channel * 2 ** (i+1)
            # self.model += [
            #     nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), # [(W−K+2P)/S]+1  256 - 3 + 1 = 254
            #     nn.InstanceNorm2d(c_out),
            #     nn.ReLU(inplace=True)
            # ]
            self.downsamp += [
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), # [(W−K+2P)/S]+1  128 - 3 + 1 = 126
                    nn.InstanceNorm2d(c_out),
                    nn.ReLU(inplace=True)
                )]

            self.downsamp[i] = self.downsamp[i].to(device)

        # residual blocks
        for i in range(num_residual_blocks):
            channel = c_out
            self.resnet += [ResnetBlock(channel)]
        
        # up sampling
        for i in range(num_up_downsampling):
            c_in = in_out_channel * 2 ** (num_up_downsampling - i)
            c_out = in_out_channel * 2 ** (num_up_downsampling - i - 1)
            
            self.upsamp += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(c_out),
                    nn.ReLU(inplace=True)
                )
            ]

            self.upsamp[i] = self.upsamp[i].to(device)

        self.end = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        self.resnet = nn.Sequential(*self.resnet)

        # move to cuda
        

    def forward(self, x):
        '''
        input: x, (B, C, dim, dim)
        output: x, (B, C, dim, dim)
        '''
        B, C, H, W = x.shape
        # U net structure
        downsample = [x]
        for i in range(len(self.downsamp)):
            x = self.downsamp[i](x)
            downsample.append(x)
        
        x = self.resnet(x)

        for i in range(len(self.upsamp)):
            x = x + downsample[-i-1]
            x = self.upsamp[i](x)
            
        
        return self.end(x) + downsample[0]
    

class Discriminator(nn.Module):

    def __init__(self, num_upsampling=3, device='cuda'):
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