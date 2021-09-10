import torch
import torch.nn as nn
import math
import pdb
from common.common_nn import T

"""
class based on the code from  :  https://github.com/d-li14/octconv.pytorch/blob/master/octconv.py

"""

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, 
                        alpha_out=0.5, stride=1, padding=0, dilation=1,
                        groups=1, bias=False, *args, **kwargs):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv1d(in_channels = int(alpha_in * in_channels), 
                                  out_channels = int(alpha_out * out_channels),
                                  kernel_size = kernel_size, stride= 1, 
                                  padding=padding, 
                                  dilation= dilation, groups=math.ceil(alpha_in * groups), 
                                  bias = bias,*args, **kwargs)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv1d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv1d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv1d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class OctaveTransposedConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5,
                        stride=1, padding=0, dilation=1,
                        groups=1, bias=False,*args, **kwargs):
        super(OctaveTransposedConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool1d(kernel_size=2, stride = 2)
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.dconv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.ConvTranspose1d(in_channels = int(alpha_in * in_channels), 
                                  out_channels = int(alpha_out * out_channels),
                                  kernel_size = kernel_size, 
                                  stride= 1, 
                                  padding=padding,
                                  output_padding = 0, 
                                  groups=math.ceil(alpha_in * groups), 
                                  bias = bias,
                                  dilation= dilation,*args, **kwargs)

        self.dconv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.ConvTranspose1d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, 0, groups,bias, dilation)
        self.dconv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.ConvTranspose1d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, 0, groups, bias, dilation)
        self.dconv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.ConvTranspose1d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, 0, math.ceil(groups - alpha_in * groups), bias, dilation)

    def forward(self, x):

        # breakpoint()
        x_h, x_l = x if type(x) is tuple else (x, None)
        x_h = self.upsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.dconv_h2h(x_h)
        x_h2l = self.dconv_h2l(self.upsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None

        if x_l is not None:
            x_l2l = self.upsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.dconv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h,x_l2lxxx
            else:
                x_l2h = self.dconv_l2h(x_l)
                x_l2h = self.downsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else: 
            return x_h2h, x_h2l

class Octave_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv=OctaveConv,
                    alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                    groups=1, bias=False, norm_layer=nn.BatchNorm1d):
        super(Octave_BatchNorm, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class OctaveBatchNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv= OctaveConv, alpha_in=0.5, 
                    alpha_out=0.5, stride=1, padding=0, dilation=1,
                    groups=1, bias=False, norm_layer=nn.BatchNorm1d, activation_layer="relu",*args,**kwargs):
        super(OctaveBatchNormActivation, self).__init__()
        # pdb.set_trace()
        self.conv = conv(in_channels, out_channels, kernel_size, alpha_in, 
                        alpha_out, stride, padding, dilation,groups, bias, *args, **kwargs)

        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = T.activation_func("leaky_relu")


    def apply(self,fn):
        # adapted the apply method form module to the OctaveBatchNormActiviation
        # This way we could initialize the nn.Conv1d/nn.ConvTranspose1d
        self._init_weight(fn)

    def _init_weight(self,fn):
        for octave_layer in self.children(): #octaveBA
            for layer in octave_layer.children(): #octave
                fn(layer)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l

def main():
    channel = [12,32,64,128,512,256]
    layers = nn.Sequential(*[
            OctaveBatchNormActivation(conv=OctaveConv, in_channels = channel[0], out_channels   = channel[1], stride=2,
                                        kernel_size = 3, padding=1, dilation=1, 
                                        activation_layer = "leaky_relu"),
            OctaveBatchNormActivation(conv=OctaveConv, in_channels = channel[1], out_channels   = channel[2], stride=2,
                                        kernel_size = 3, padding=1, dilation=1, 
                                        activation_layer = "leaky_relu")])
    pdb.set_trace()
    dummy = torch.randn(10, 6, 4096)
    x_h, x_l = layers(dummy)
    x_l = layer7(x_l)

if __name__ == '__main__':
    main()