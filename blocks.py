# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:29:06 2020

@author: T_ESTIENNE
"""
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_activation(activation_type):
    activation_type = activation_type.lower()

    if activation_type == 'prelu':
        return nn.PReLU()
    elif activation_type == 'leaky':
        return nn.LeakyReLU(inplace=True)
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'softmax':
        return nn.Softmax()
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'relu':
        return nn.ReLU(inplace=True)
    else:
        return Identity()

class UpPoolingConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, activation_type='prelu', 
                 instance_norm=False, batch_norm=False):
        
        super(UpPoolingConvolution, self).__init__()
        
        self.instance_norm = instance_norm or batch_norm
        
        self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                       kernel_size=2, padding=0, stride=2)

        self.activation = get_activation(activation_type)
        
        if instance_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        elif batch_norm:
            self.norm = nn.BatchNorm3d(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        out = self.activation(x)
        return out
    
class DownPoolingConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, activation_type='prelu', 
                 instance_norm=False, batch_norm=False):
        
        super(DownPoolingConvolution, self).__init__()
        
        self.instance_norm = instance_norm or batch_norm
        
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=2, padding=0, stride=2)

        self.activation = get_activation(activation_type)
        
        if instance_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        elif batch_norm:
            self.norm = nn.BatchNorm3d(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        out = self.activation(x)
        return out

class Convolution(nn.Module):

    def __init__(self, in_channels, out_channels, activation_type='prelu',
                 instance_norm=False, batch_norm=False):
        super(Convolution, self).__init__()
        
        self.instance_norm = instance_norm or batch_norm

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              stride=1)
 
        self.activation = get_activation(activation_type)
        
        if instance_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        elif batch_norm:
            self.norm = nn.BatchNorm3d(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        out = self.activation(x)
        return out


def _make_nConv(in_channels, out_channels, activation, instance_norm,
                batch_norm, nb_Conv):
    layers = []
    layers.append(Convolution(in_channels, out_channels, activation, 
                              instance_norm, batch_norm))
    for _ in range(nb_Conv-1):
        layers.append(Convolution(out_channels, out_channels,
                                  activation, instance_norm=instance_norm,
                                  batch_norm=batch_norm))
    
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 activation_type='leaky', instance_norm=False, batch_norm=False,
                 nb_Conv=1
                 ):

        super(ConvBlock, self).__init__()

        self.conv_pool = DownPoolingConvolution(in_channels, out_channels, 
                                                activation_type, 
                                                instance_norm, batch_norm
                                                )


        self.nConvs = _make_nConv(out_channels, out_channels, activation_type,
                                  instance_norm, batch_norm, nb_Conv)

    def forward(self, x):
        
        down = self.conv_pool(x)
        out = self.nConvs(down)

        return out


class DeconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 activation_type='leaky',
                 instance_norm=False, batch_norm=False,
                 nb_Conv=1):

        super(DeconvBlock, self).__init__()

        self.conv_tr = UpPoolingConvolution(in_channels, out_channels,
                                            activation_type=activation_type,
                                            instance_norm=instance_norm,
                                            batch_norm=batch_norm)

        self.nConvs = _make_nConv(in_channels, out_channels, activation_type,
                                  instance_norm, batch_norm, nb_Conv)

    def forward(self, x, skip_x=None):
        
        up = self.conv_tr(x)
        cat = torch.cat((up, skip_x), 1)
        out = self.nConvs(cat)
        
        return out


class InputBlock(nn.Module):

    def __init__(self, n_channels, input_channels,
                 activation_type='leaky'):

        super(InputBlock, self).__init__()

        self.add_conv = nn.Conv3d(in_channels=input_channels,
                                  out_channels=n_channels,
                                  kernel_size=3, padding=1, stride=1)


        self.conv = nn.Conv3d(in_channels=n_channels,
                              out_channels=n_channels,
                              kernel_size=3,
                              padding=1,
                              stride=1)

        self.activation = get_activation(activation_type)


    def forward(self, x):
        
        x = self.activation(self.add_conv(x))
        out = self.activation(self.conv(x))
        out = torch.add(out, x)
        
        return out


class Decoder(nn.Module):

    def __init__(self, pool_blocks, channels, out_channels, last_activation,
                 activation_type='leaky',
                 instance_norm=False, batch_norm=False,
                 nb_Convs=[1, 1, 1, 1, 1],
                 freeze_registration=False, zeros_init=False,
                 deep_supervision=False):

        super(Decoder, self).__init__()

        self.conv_list = nn.ModuleList()
        self.deep_supervision = deep_supervision

        for i in range(0, pool_blocks):
            self.conv_list.append(DeconvBlock(channels[-i-1], channels[-i-2],
                                         activation_type, instance_norm,
                                         batch_norm, nb_Convs[-i-1]))


        self.last_conv = nn.Conv3d(in_channels=channels[-pool_blocks-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1)

        self.last_activation = get_activation(last_activation)
        
        if freeze_registration:
            for param in self.last_conv.parameters():
                    param.requires_grad = False
            
        if zeros_init:
            torch.nn.init.zeros_(self.last_conv.weight)
            torch.nn.init.zeros_(self.last_conv.bias)

    def forward(self, skip_x):
        
        if self.deep_supervision:
            pred_list = []

        out = skip_x[-1]
        
        for i, conv in enumerate(self.conv_list):
            if self.deep_supervision:
                pred_list.append(out)
            out = conv(out, skip_x[-i-2])
        
        out = self.last_conv(out)
        out = self.last_activation(out)

        if self.deep_supervision:
            pred_list.append(out) # Last one is the final prediction
            return pred_list
        else:   
            return out


class Encoder(nn.Module):

    def __init__(self, pool_blocks, channels, 
                 activation_type, input_channel=4,
                 instance_norm=False, batch_norm=False,
                 nb_Convs=[1, 1, 1, 1, 1]):
        
        super(Encoder, self).__init__()

        self.input = InputBlock(channels[0], input_channel,
                                activation_type)
        self.conv_blocks = nn.ModuleList()
        
        for i in range(pool_blocks):

            self.conv_blocks.append(ConvBlock(channels[i], channels[i+1],
                                              activation_type, 
                                              instance_norm, batch_norm,
                                              nb_Convs[i]
                                              )
                                    )

    def forward(self, x):
        
        
        skip_x = []
        skip_x.append(self.input(x))

        for conv in self.conv_blocks:
            skip_x.append(conv(skip_x[-1]))

        return skip_x


class DeepSupervisionBlock(nn.Module):
        
    def __init__(self, pool_blocks, channels, out_channels,
                 last_activation):

        super(DeepSupervisionBlock, self).__init__()
        
        self.convs_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        
        for i in range(pool_blocks):
            self.convs_list.append(nn.Conv3d(in_channels=channels[-i-1],
                                             out_channels=out_channels,
                                             kernel_size=3, padding=1, 
                                             stride=1))
            self.upsample_list.append(nn.Upsample(scale_factor=2**(pool_blocks-i)))
            
        
        self.last_activation = get_activation(last_activation)
    
    def forward(self, x_list):
        
        pred_list = []

        for i, x in enumerate(x_list[:-1]):
            
            out = self.convs_list[i](x) # Change number of filter
            up = self.last_activation(self.upsample_list[i](out)) # Upsample
            pred_list.append(up)
        
        pred_list.append(x_list[-1])
        return pred_list