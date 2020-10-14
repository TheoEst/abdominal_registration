# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:32:53 2020

@author: T_ESTIENNE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abdominal_registration import blocks

def integral3DImage(deformable):
    
    img_size = deformable.shape[-3:]
    
    x_s = (4*torch.cumsum(deformable[:, 0, :, :], dim=1) - 2)/ (img_size[0] - 1) - 1
    y_s = (4*torch.cumsum(deformable[:, 1, :, :], dim=2) - 2)/ (img_size[1] - 1) - 1
    z_s = (4*torch.cumsum(deformable[:, 2, :, :], dim=3) - 2)/ (img_size[2] - 1) - 1

    out = torch.stack([z_s, y_s, x_s], dim=1)  # Shape B*3*H*W*D

    return out


def diffeomorphic3D(moving, integrated_grid=None):

    integrated_grid = integrated_grid.permute(0, 2, 3, 4, 1)

    # Deformed moving image
    deformed_img = F.grid_sample(moving, integrated_grid)

    return deformed_img
    

class FrontiersNet(nn.Module):

    def __init__(self,
                 channel_multiplication=4,
                 pool_blocks=4,
                 channels=[4, 8, 16, 32, 64, 128, 256],
                 cross_entropy=False,
                 activation_type='relu',
                 last_activation=None,
                 instance_norm=False,
                 batch_norm=False,
                 nb_Convs=[1, 1, 1, 1, 1, 1, 1],
                 freeze_registration=False,
                 zeros_init=False,
                 symmetric_training=False,
                 multi_windows=False,
                 deep_supervision=False,
                 keep_all_label=False,
                 ):

        super(FrontiersNet, self).__init__()

        channels = channels[:pool_blocks+1]
        channels = [int(channel * channel_multiplication)
                    for channel in channels]
        nb_Convs = nb_Convs[:pool_blocks]
        
        input_channel = 3 if multi_windows else 1
        self.deep_supervision = deep_supervision
        self.symmetric_training = symmetric_training

        kwargs = {'pool_blocks': pool_blocks, 'channels': channels,
                  'activation_type': activation_type,
                  'instance_norm': instance_norm, 'batch_norm' :batch_norm,
                  'nb_Convs':nb_Convs,
                 }
                
        # Encoder
        self.encoder = blocks.Encoder(input_channel=input_channel,
                                      **kwargs)
        
        # Decoder
        registration_decoder_out_channels = 3
        

        self.registration_decoder = blocks.Decoder(out_channels=registration_decoder_out_channels,
                                                   last_activation='sigmoid',
                                                   freeze_registration=freeze_registration,
                                                   zeros_init=zeros_init,
                                                   deep_supervision=deep_supervision,
                                                   **kwargs)
            
        self.merge_operator = torch.sub
        
        if deep_supervision:
            self.deep_blocks = blocks.DeepSupervisionBlock(pool_blocks, channels,
                                                           registration_decoder_out_channels,
                                                           last_activation='sigmoid')
        

    def forward(self, moving, reference):
        
        moving_encoding = self.encoder(moving)
        reference_encoding = self.encoder(reference)
        
            
        mergin_moving = self.merging(moving_encoding, reference_encoding)
       
        (deformable_grid, integrated_grid, 
         deformed_img) = self.apply_deformation(moving, mergin=mergin_moving)
                        
        pred_sample = [(deformable_grid, integrated_grid, deformed_img)]
        
        if self.symmetric_training:
            mergin_reference = self.merging(reference_encoding, moving_encoding)
            
            (deformable_grid, integrated_grid,
             deformed_img) = self.apply_deformation(reference, mergin=mergin_reference)
                 
            pred_sample.append( (deformable_grid, integrated_grid, deformed_img) )
            

        return pred_sample
    
    def apply_deformation(self, source, mergin):
        
        deformable_grid = self.registration_decoder(mergin)
            
        if self.deep_supervision:
            deformable_grid =  self.deep_blocks(deformable_grid)
            integrated_grid = [integral3DImage(grid) for grid in deformable_grid]
            
            deformed_img = [diffeomorphic3D(source, grid) for grid in integrated_grid]

            deformable_grid = deformable_grid[-1]
            integrated_grid = integrated_grid[-1]
        else:
            integrated_grid = integral3DImage(deformable_grid)
            deformed_img = diffeomorphic3D(source, integrated_grid)
                    
        
        return deformable_grid, integrated_grid, deformed_img
    
    def merging(self, encoding_x, encoding_y):
        
        merged_encoding = []
        
        for x, y in zip(encoding_x, encoding_y):
            merged_encoding.append( self.merge_operator(x, y))
        
        return merged_encoding


if __name__ == '__main__':
    model = FrontiersNet(channel_multiplication=8)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

