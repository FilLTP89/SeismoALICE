import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from common.common_nn import T


class ConvBlock(nn.Module):
    """docstring for model_type"""
    def __init__(self, in_channels, out_channels, activation_function='relu',
                    down=True, use_act = True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode='reflect', **kwargs) 
            if down
            else nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            T.activation_func(activation_function) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module): 
    def __init__(self, channels, activation_function = 'relu', *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels = channels, out_channels = channels, kernel_size=3, padding = 1,\
                    activation_function = activation_function, *args, **kwargs), 
            ConvBlock(in_channels = channels, out_channels = channels, use_act = False, kernel_size = 3, padding = 1,\
                    activation_function = activation_function, *args, **kwargs)
            )

    def forward(self, x): 
        return x + self.block(x)


if __name__ == "__main__":
    res = ResidualBlock(channels=64, activation_function='selu')
    x   = torch.randn(10,64,64)
    y   = res(x)

    print(x.shape)
    print(y.shape)

