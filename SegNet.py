# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:16:42 2020

@author: t_estienne
"""
import torch.nn as nn
from abdominal_registration import blocks

class SegNet(nn.Module):

    def __init__(self,
                 channel_multiplication=4,
                 pool_blocks=4,
                 channels=[4, 8, 16, 32, 64, 128, 256],
                 activation_type='leaky',
                 instance_norm=False,
                 batch_norm=False,
                 nb_Convs=[1, 1, 1, 1, 1, 1, 1],
                 ):

        super(SegNet, self).__init__()

        channels = channels[:pool_blocks+1]
        channels = [int(channel * channel_multiplication)
                    for channel in channels]
        nb_Convs = nb_Convs[:pool_blocks]
        
        input_channel = 1
        segmentation_out_channels = 12
            

        kwargs = {'pool_blocks': pool_blocks, 'channels': channels,
          'activation_type': activation_type, 'nb_Convs':nb_Convs,
          'instance_norm': instance_norm, 'batch_norm' :batch_norm,
          }
                
        # Encoder
        self.encoder = blocks.Encoder(input_channel=input_channel,
                                      **kwargs)

        # Decoder
        seg_last_activation = 'softmax'
        
        self.segmentation_decoder = blocks.Decoder(out_channels=segmentation_out_channels, 
                                                   last_activation=seg_last_activation,
                                                   **kwargs)

    def forward(self, ct):
        
        z = self.encoder(ct)
        mask = self.segmentation_decoder(z)
        
        return mask
    

    
if __name__ == '__main__':
    model = SegNet(channel_multiplication=4)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

