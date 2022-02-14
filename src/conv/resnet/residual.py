import torch
import torch.nn as nn
from functools import partial
import pdb


"""
class based in the implemented code from :  
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
"""

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['tanh', nn.Tanh()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=1.0, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class block_3x3(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, conv = partial(nn.ConvTranspose1d),
        identity_downsample=None, stride=1, activation = 'leaky_relu',*args, **kwargs
    ):
        super(block_3x3, self).__init__()
        self.expansion = 4
        self.conv1 = conv(in_channels, intermediate_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = conv(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            *args, **kwargs
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.conv3 = conv(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.activation = activation_func(activation)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.activation(x)
        return x


class block_2x2(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, conv = partial(nn.ConvTranspose1d),
        identity_downsample=None, stride=1, activation = 'leaky_relu',*args, **kwargs
    ):
        super(block_2x2, self).__init__()
        self.expansion = 4

        self.conv1 = conv(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            *args, **kwargs
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = conv(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.activation = activation_func(activation)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.activation(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks, intermediate_channels,
                    stride, conv = partial(nn.Conv1d), block=block_3x3):
        super(ResNet,self).__init__()

        self.identity_downsample    = None
        self.num_residual_blocks    = num_residual_blocks
        self.intermediate_channels  = intermediate_channels
        self.in_channels = in_channels
        self.layers = []
        self.stride = stride
        self.block  = block
        self.conv   = conv

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
    def _make_layer(self,*args, **kwargs):
        if self.stride != 1 or self.in_channels != self.intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                self.conv(
                    self.in_channels,
                    self.intermediate_channels * 4,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                    *args, **kwargs
                ),
                nn.BatchNorm1d(self.intermediate_channels * 4),
            )
        self.layers.append(
            self.block(self.in_channels, self.intermediate_channels, self.conv, 
            identity_downsample, self.stride, *args, **kwargs)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = self.intermediate_channels * 4
        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(self.num_residual_blocks - 1):
            self.layers.append(self.block(self.in_channels, self.intermediate_channels))

        self.layers = nn.Sequential(*self.layers)
        return self.layers, self.in_channels


class EncoderResnet(nn.Module): 
    def __init__(self, image_channels, channels,layers, block = block_3x3, *args, **kwargs): 
        super().__init__()
        # pdb.set_trace()
        self.in_channels= image_channels
        self.channels   = channels
        self.conv1      = nn.Conv1d(image_channels, self.channels[0], kernel_size=7,
                             stride=2, padding=3, bias=False
                        )
        self.bn1        = nn.BatchNorm1d(64)
        self.relu       = activation_func('relu')
        self.tanh       = activation_func('tanh')
        self.conv2      = nn.Conv1d(in_channels = self.channels[0], 
                                out_channels= self.channels[0],
                                kernel_size=3, stride=2, padding=1
                        )
        self.layers      = layers
        self.network     = []

        # Essentially the entire ResNet architecture are in these 4 lines below
        _layer, self.in_channels = ResNet(block, self.in_channels, self.layers[0],
                    intermediate_channels=self.channels[0], stride=1, 
                    conv = partial(nn.Conv1d)
                    )._make_layer(*args, **kwargs)
        self.network.append(_layer)
       
        for layer, channel in zip(self.layers[1:],self.channels[1:]):
            _layer , self.in_channels = ResNet(block, self.in_channels, layer,
                    intermediate_channels= channel, stride=2, 
                    conv = partial(nn.Conv1d)
                    )._make_layer(*args, **kwargs)
            self.network.append(_layer)

        self.network = nn.Sequential(*self.network)
        #ending layer
        self.conv3= nn.Conv1d(in_channels = self.channels[-1]*4, out_channels= 32,
                    kernel_size = 3, stride = 1, padding = 1
                    )

    def _convolution(self): 
        return partial(nn.Conv1d)

    def forward(self,x) :
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.network(x)

        x = self.conv3(x)
        x = self.tanh(x)

        return x

        
class DecoderResnet(nn.Module):
    def __init__(self, image_channels, channels, layers, block=block_3x3, *args, **kwargs):
        super().__init__()
        self.in_channels = image_channels
        self.channels    = channels
        self.conv        = self._convolution()
        self.leaky_relu  = activation_func('leaky_relu')
        self.layers      = layers
        self.conv1       = nn.ConvTranspose1d(image_channels, self.channels[0], 
                            kernel_size=7, stride=2, padding=3, output_padding=1,bias=False)
        self.bn1         = nn.BatchNorm1d(self.channels[0])
        self.conv2       = nn.ConvTranspose1d(in_channels = self.channels[0],out_channels=self.channels[0],
                             kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        self.network   = []

        
        _layer, self.in_channels = ResNet(block=block, in_channels=self.in_channels,
                    num_residual_blocks=self.layers[0],
                    intermediate_channels=self.channels[0], stride=1, 
                    conv = partial(nn.ConvTranspose1d)
                    )._make_layer(*args,**kwargs)
        self.network.append(_layer)
       
        for layer, channel in zip(self.layers[1:],self.channels[1:]):
            _layer , self.in_channels = ResNet(block=block, in_channels=self.in_channels,
                    num_residual_blocks=layer,
                    intermediate_channels= channel, stride=2, 
                    conv = partial(nn.ConvTranspose1d)
                    )._make_layer(output_padding=1)
            self.network.append(_layer)
        
        self.network = nn.Sequential(*self.network)
        
        self.conv3 = nn.ConvTranspose1d(in_channels = self.channels[-1]*4, out_channels= 3,
                     kernel_size = 3, stride = 2, padding = 1, output_padding=1) 

    def _convolution(self): 
        return partial(nn.ConvTranspose1d)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.network(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)

        return x

def test():
    # net = ResNet50(img_channel=3, num_classes=1000)
    # y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    # print(y.size())

    # net =  EncoderResnet(image_channels = 6, channels = [64,128, 256, 512, 512], layers = [3,4,6,3,2])
    # t   = torch.randn(10, 6, 4096)
    # print(t.shape)
    # y   =  net(t)
    # print(y.shape)
    breakpoint()
    t   = torch.randn(10, 16, 128)
    print(t.shape)
    net = DecoderResnet(image_channels = 16, channels = [64, 128, 256], layers = [2,2,2], block=block_2x2)
    x   = net(t)
    print(x.shape)

if __name__ == '__main__':
    test()
