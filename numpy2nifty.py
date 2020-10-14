# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:24:29 2020

@author: T_ESTIENNE
"""
import numpy as np
import SimpleITK as sitk

def recover_from_crop(crop_array, output_size, orig_size):
    
    depth, height, width = crop_array.shape

    if len(orig_size) == 4:
        _, depth, height, width = orig_size
    else:
        depth, height, width = orig_size # Oasis only 3D, no modality
    
    
    if depth == output_size[0]:
        depth_min = 0
        depth_max = depth
    else:
        depth_min = int((depth - output_size[0])/2)
        depth_max = -(depth - output_size[0] - depth_min)

    if height == output_size[1]:
        height_min = 0
        height_max = height
    else:
        height_min = int((height - output_size[1])/2)
        height_max = -(height - output_size[1] - height_min)

    if width == output_size[2]:
        width_min = 0
        width_max = width
    else:
        width_min = int((width - output_size[2])/2)
        width_max = -(width - output_size[2] - width_min)

    orig_array = np.zeros(orig_size)
    
    if len(orig_size) == 4:
        orig_array[:,  depth_min:depth_max, height_min:height_max, 
                   width_min:width_max] = crop_array
        
    elif len(orig_size) == 3:
        orig_array[depth_min:depth_max, height_min:height_max, 
                   width_min:width_max] = crop_array

    return orig_array

def numpy2nifty(array, sitk_img, args):

    orig_size = sitk_img.GetSize()
    orig_size = [orig_size[2], orig_size[1], orig_size[0]]
    
    array = recover_from_crop(array, args.crop_size, orig_size)

    img = sitk.GetImageFromArray(array)
    
    img.SetDirection(sitk_img.GetDirection())
    img.SetSpacing(sitk_img.GetSpacing())
    img.SetOrigin(sitk_img.GetOrigin())
    
    return img
