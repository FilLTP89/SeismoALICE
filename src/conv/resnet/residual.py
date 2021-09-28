import torch
import torch.nn as nn
from functools import partial
import pdb


"""
class based in the code form :  
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

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, conv = partial(nn.ConvTranspose1d),
        identity_downsample=None, stride=1, activation = 'leaky_relu',*args, **kwargs
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = conv(in_channels, intermediate_channels, 
            kernel_size=1, stride=1, padding=0, bias=False,*args,**kwargs
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
            bias=False,
            *args, **kwargs
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


class ResNet(nn.Module):

    def __init__(self, block, num_residual_blocks, intermediate_channels,
                     stride, conv = partial(nn.Conv1d), *args, **kwargs):
        super(ResNet,self).__init__()
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                conv(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    *args, **kwargs
                ),
                nn.BatchNorm1d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, conv, identity_downsample, stride, *args, **kwargs)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return x


class EncoderResnet(ResNet): 
    def __init__(self, layers, image_channels, block = block, *args, **kwargs): 
        super().__init__()
        # pdb.set_trace()
        self.in_channels = 64
        self.conv1       = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1         = nn.BatchNorm1d(64)
        self.relu        = activation_func('relu')
        self.tanh        = activation_func('tanh')
        self.conv2       = nn.Conv1d(in_channels = 64, out_channels=64 ,kernel_size=3, stride=2, padding=1)
        

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = ResNet(block, layers[0], intermediate_channels=64, stride=1, 
            conv = partial(nn.Conv1d),*args, **kwargs)

        self.layer2 = ResNet(block, layers[1], intermediate_channels=128, stride=2, 
            conv = partial(nn.Conv1d),*args, **kwargs)

        self.layer3 = ResNet(block, layers[2], intermediate_channels=256, stride=2, 
            conv = partial(nn.Conv1d),*args, **kwargs)

        self.layer4 = ResNet(block, layers[3], intermediate_channels=512, stride=2, 
            conv = partial(nn.Conv1d),*args, **kwargs)
        
        self.layer5 = ResNet(block, layers[4], intermediate_channels=512, stride=2, 
            conv = partial(nn.Conv1d),*args, **kwargs)

        #ending layer
        self.conv3= nn.Conv1d(in_channels = 512*4, out_channels= 32, kernel_size = 3, stride = 1, padding = 1)

    def forward(self,x) :
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv3(x)
        x = self.tanh(x)

        return x

        
class DecoderResnet(nn.Module, ResNetInterface):
    def __init__(self, layers, image_channels, block=block, *args, **kwargs):
        super().__init__()
        # pdb.set_trace()
        self.in_channels = 64
        self.conv = self._convolution()
        self.conv1       = nn.ConvTranspose1d(image_channels, 64, kernel_size=7, stride=2, 
                            padding=3, output_padding=1,bias=False)
        self.bn1         = nn.BatchNorm1d(64)
        self.leaky_relu  = activation_func('leaky_relu')
        self.conv2       = nn.ConvTranspose1d(in_channels = 64, out_channels=64 ,kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1, conv=partial(nn.ConvTranspose1d),*args, **kwargs
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2, conv=partial(nn.ConvTranspose1d),*args, **kwargs
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=128, stride=2, conv=partial(nn.ConvTranspose1d),*args, **kwargs
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2, conv=partial(nn.ConvTranspose1d),*args, **kwargs
        )

        self.layer5 = self._make_layer(
            block, layers[4], intermediate_channels=512, stride=2, conv=partial(nn.ConvTranspose1d),*args, **kwargs
        )

        self.conv3 = nn.ConvTranspose1d(in_channels = 512*4, out_channels= 3, kernel_size = 34, 
            stride = 1, padding = 1) 

    def _convolution(self): 
        return partial(nn.ConvTranspose1d)

    def forward(self,x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        return x

def test():
    # net = ResNet50(img_channel=3, num_classes=1000)
    # y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    # print(y.size())
    pdb.set_trace()
    net =  EncoderResnet(image_channels = 6, layers = [3,4,6,3,2])
    t   = torch.randn(10, 6, 4096)
    print(t.shape)
    y   =  net(t)
    print(y.shape)


    t   = torch.randn(10, 64, 64)
    print(t.shape)
    net = DecoderResnet(image_channels = 64, layers = [3,4,6,3,2])
    x   = net(t)
    print(x.shape)

if __name__ == '__main__':
    test()
