import torch.nn as nn


class Generator(nn.Module):   
    """
    This class implements a Generator inspired by DCGAN.
    Uses Fractionally Strided Convolutions to ensure that the training learns it own upsampling layers.
    Additionally ReLu activations are applied between each layer + Batchnorm for more stable training. 
    In the output Layer one can find a tanh-Activation.
    """
    
    def __init__(self, z_dim: int = 100):
        """
        Implementation of Generator with aspects from the paper which introduced DCGAN (Deep Convolutional GAN's)
        
        Normally this class takes a 100 dimensional Z-Vector as input and upsamples it to an image of 1024x1024. If another
        dimensionality is being chosen then it can not be guranteed to match the with the Discrimnator anymore.
        
        params:
        -----------
        z_dim: (int)
            Dimensionality of the Latent Vector
        """
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            # Input Latent Vector
            nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=1024, 
                               kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            # 4x4
            # Adjustment of this layer stride of 4 -> 16x16 out
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, 
                               kernel_size=(4, 4), stride=4, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            # 16x16
            nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # output Size = 32
            nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # output Size = 64
            nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # output Size = 128
            nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # output Size = 256
            nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            # output Size = 512
            nn.ConvTranspose2d(in_channels=16, out_channels=3, 
                               kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.Tanh(),
            # Out = 1024
        )

    def forward(self, input):
        return self.gen(input)
    
class Discriminator(nn.Module):
    """
    This class implements the Discriminator inspired by the DCGAN. 
    Uses LeakyRelu und Sigmoid activation in the end as well as Batchnorm bt. each layer for a more stable training.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input 1024x1024 per channel
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x512
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x256
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x128
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Resulting in one output neuron for BCE-Loss
        )

    def forward(self, input):
        return self.main(input)
