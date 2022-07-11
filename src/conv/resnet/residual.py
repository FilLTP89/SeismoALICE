import torch
import torch.nn as nn
from functools import partial
from core.net.basic_model import BasicModel
import pdb


"""
class based in the implemented code from :  
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
"""

def activation_func(activation, slope=0.1):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['tanh', nn.Tanh()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=slope, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class IStrategyConvolution(object):
    def __init__(self):
        super(IStrategyConvolution, self).__init__()
        pass

    def expansion(*args, **kwargs):
        raise NotImplementedError

    def functions(*args, **kwargs):
        raise NotImplementedError



class ConvolutionTools(IStrategyConvolution):
    def __init__(self):
        super(ConvolutionTools,self).__init__()
        pass

    def expansion(self,channels, expansion):
        return channels*expansion

    def functions(self,*args, **kwargs):
        func1, func2 =  activation_func('leaky_relu',*args, **kwargs), activation_func('leaky_relu',*args, **kwargs)
        return func1, func2

    def padding(self):
        return 1

class ConvTransposeTools(IStrategyConvolution):
    """docstring for ConvTransposeTools"""
    def __init__(self):
        super(ConvTransposeTools, self).__init__()
        pass
    
    def expansion(self,channels, expansion):
        return channels//expansion

    def functions(self,*args, **kwargs):
        func1, func2 =  activation_func('relu',*args, **kwargs), activation_func('relu',*args, **kwargs)
        return func1, func2
    def padding(self):
        return 1



class block_3x3(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, conv_tools, conv = partial(nn.ConvTranspose1d),
        identity_downsample=None, stride=1, activation = 'leaky_relu',*args, **kwargs
    ):
        super(block_3x3, self).__init__()
        self.conv_tools = conv_tools
        self._expansion  = 4
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
            self.conv_tools.expansion(intermediate_channels, self._expansion),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(self.conv_tools.expansion(intermediate_channels,self._expansion))
        self.activation1,self.activation2 = self.conv_tools.functions()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.activation2(x)
        return x


class block_2x2(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, conv_tools, conv = partial(nn.ConvTranspose1d),
        identity_downsample=None, stride=1, activation = 'leaky_relu', *args, **kwargs
    ):
        super(block_2x2, self).__init__()
        self._expansion  = 2
        self.conv_tools = conv_tools
        
        self.conv1 = conv(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
            *args, **kwargs
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = conv(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.activation1, self.activation2 = self.conv_tools.functions()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.activation2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks, intermediate_channels,
                    stride, conv = partial(nn.Conv1d), block=block_3x3, 
                    conv_tools=ConvolutionTools
                ):
        super(ResNet,self).__init__()
        
        self.identity_downsample    = None
        self.num_residual_blocks    = num_residual_blocks
        self.intermediate_channels  = intermediate_channels
        self.in_channels = in_channels
        self.layers = []
        self.stride = stride
        self.block  = block
        self.conv   = conv
        self.conv_tools = conv_tools
        self._expansion   = 4 if self.block == block_3x3 else 2

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
    def _make_layer(self,*args, **kwargs):
        _out_channel = self.in_channels
        if self.stride != 1 or self.in_channels != self.conv_tools.expansion(self.intermediate_channels,self._expansion):
            identity_downsample = nn.Sequential(
                self.conv(
                    self.in_channels,
                    self.intermediate_channels,
                    kernel_size=3,
                    stride=self.stride,
                    bias=False,
                    padding=self.conv_tools.padding(),
                    *args, **kwargs
                ),
                nn.BatchNorm1d(self.intermediate_channels),
            )
        self.layers.append(
            self.block(in_channels=self.in_channels, 
                    intermediate_channels=self.intermediate_channels, 
                    conv_tools=self.conv_tools, conv=self.conv, 
                    identity_downsample = identity_downsample, stride=self.stride,*args,**kwargs)
            )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = self.intermediate_channels # self.conv_tools.expansion(self.intermediate_channels,self._expansion)
        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(self.num_residual_blocks - 1):
            self.layers.append(self.block(in_channels=self.in_channels, 
                    intermediate_channels=self.intermediate_channels, 
                    conv_tools=self.conv_tools, conv=self.conv
                )
            )

        self.layers = nn.Sequential(*self.layers)
        return self.layers, self.in_channels

class ResidualContainer(BasicModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_val    = 0
        self._models        = []
    
    def __len__(self):
        return len(self._models)
    
    def __iter__(self):
        return iter(self._models)
    
    def __getitem__(self,idx):
        return self._models[idx]
    
    def __next__(self):
        if self.current_val>=len(self._models):
            self.current_val=0
            raise StopIteration
        _model = self._models[self.current_val]
        self.current_val +=1
        return _model

class EncoderResnet(ResidualContainer): 
    def __init__(self, in_signals_channels, out_signals_channels, channels,layers, 
                        block = block_3x3, *args, **kwargs): 
        super().__init__()
        # pdb.set_trace()
       
        self.in_channels= in_signals_channels
        self.conv_tools = self._convolution_tools()
        self.channels   = channels
        self.layers     = layers
        self.conv1      = nn.Conv1d(in_signals_channels, self.channels[0], kernel_size=7,
                             stride=1, padding=3, bias=False
                        )
        self.bn1        = nn.BatchNorm1d(self.channels[0])
        self.leaky_relu1, self.leaky_relu2  = self.conv_tools.functions()
        self.conv2      = nn.Conv1d(in_channels = self.channels[0], 
                            out_channels= self.channels[0],
                            kernel_size=3, stride=2, padding=1, bias = False
                        )
        
        self.network    = []
        self._expansion = 4 if isinstance(block, block_3x3) else 2

        # Essentially the entire ResNet architecture are in these 4 lines below
        _layer, self.in_channels = ResNet(block=block, in_channels=self.channels[0], 
                    num_residual_blocks=self.layers[0],
                    intermediate_channels=self.channels[0], stride=1, 
                    conv = partial(nn.Conv1d), conv_tools= self.conv_tools
                    )._make_layer()
        self.network.append(_layer)
        
        for layer, channel in zip(self.layers[1:],self.channels[1:]):
            _layer , self.in_channels = ResNet(block=block,in_channels=self.in_channels, 
                    num_residual_blocks=layer,
                    intermediate_channels= channel, stride=2, 
                    conv = partial(nn.Conv1d), conv_tools=self.conv_tools
                    )._make_layer()
            self.network.append(_layer)

        self.network = nn.Sequential(*self.network)
        #ending layer
        self.conv3 = nn.Conv1d(in_channels = self.channels[-1], 
                    out_channels= out_signals_channels,
                    kernel_size = 3, stride = 1, padding = 1, bias= False
                )
        
        self._models = [self.conv1, self.bn1, self.leaky_relu1, self.conv2, self.network, self.conv3]

    def _convolution_tools(self): 
        return ConvolutionTools()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)

        x = self.network(x)

        x = self.conv3(x)
        # x = self.leaky_relu2(x)

        return x

        
class DecoderResnet(ResidualContainer):
    def __init__(self, in_signals_channels, out_signals_channels, channels, layers, 
                        block=block_3x3, *args, **kwargs):
        super().__init__()
        
        self.current_val=0
        self.in_channels= in_signals_channels
        self.channels   = channels
        self.conv_tools = self._convolution_tools()
        self.relu, _    = self.conv_tools.functions()
        self.tanh       = activation_func('tanh')
        self.layers     = layers
        self.conv1      = nn.ConvTranspose1d(in_signals_channels, self.channels[0], 
                            kernel_size=7, stride=1, padding=3, output_padding=0,bias=False)
        self.bn1        = nn.BatchNorm1d(self.channels[0])
        self.conv2      = nn.ConvTranspose1d(in_channels = self.channels[0],out_channels=self.channels[0],
                             kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        self.network    = []
        self._expansion = 4 if isinstance(block, block_2x2) else 2
        
        _layer, self.in_channels = ResNet(block=block, in_channels=self.channels[0],
                        num_residual_blocks=self.layers[0],
                        intermediate_channels=self.channels[0], stride=1, 
                        conv = partial(nn.ConvTranspose1d), conv_tools=self.conv_tools
                    )._make_layer()
        self.network.append(_layer)
       
        for layer, channel in zip(self.layers[1:],self.channels[1:]):
            _layer, self.in_channels = ResNet(block=block, in_channels= self.in_channels,
                        num_residual_blocks=layer,intermediate_channels= channel, stride=2, 
                        conv = partial(nn.ConvTranspose1d), conv_tools=self.conv_tools
                    )._make_layer(output_padding=1)
            self.network.append(_layer)
        
        self.network= nn.Sequential(*self.network)
        
        self.conv3  = nn.ConvTranspose1d(in_channels = self.channels[-1], out_channels=out_signals_channels, 
                        kernel_size = 3, stride = 1, padding = 1, output_padding=0)

        self._models = [self.conv1, self.bn1, self.relu, self.conv2, self.network, self.conv3, self.tanh]

    def _convolution_tools(self): 
        return ConvTransposeTools()

    def forward(self,x):
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.network(x)
        x = self.conv3(x)

        x = self.tanh(x)

        return x

def test():
    # net = ResNet50(img_channel=3, num_classes=1000)
    # y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    # print(y.size())
    breakpoint()
    net = EncoderResnet(in_signals_channels = 6, out_signals_channels=1,channels = [16,32,64], layers = [2,2,2], block = block_2x2)
    t   = torch.randn(10, 32, 1024)
    print(t.shape)
    y   =  net(t)
    print(y.shape)
    print('number of parameters : {:,}'.format(net.number_parameter))
    
    t   = torch.randn(10, 16, 128)
    print(t.shape)
    net = DecoderResnet(in_signals_channels = 16, 
                out_signals_channels=3,
                channels = [128, 64, 32], 
                layers = [2,2,2], block=block_2x2
            )
    x   = net(t)
    print(x.shape)
    print('number of parameters : {:,}'.format(net.number_parameter))

if __name__ == '__main__':
    test()
