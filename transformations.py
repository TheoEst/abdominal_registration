# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:40:07 2020

@author: T_ESTIENNE
"""
import numpy as np
import random 

def crop(array, size):

    depth_min, depth_max, height_min, height_max, width_min, width_max = size

    if len(array.shape) == 3:
        crop_array = array[depth_min:depth_max,
                           height_min:height_max,
                           width_min:width_max,
                           ]
    elif len(array.shape) == 4:
        crop_array = array[:, depth_min:depth_max,
                           height_min:height_max,
                           width_min:width_max,
                           ]
    else:
        print(array.shape)
        raise ValueError

    return crop_array


class Crop(object):

    def __init__(self, output_size, dim=3):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = dim * (output_size,)
        else:
            assert len(output_size) == dim
            self.output_size = output_size
        
    def __call__(self, sample):
        pass

def center_crop_indices(img, output_size):
    
    _, depth, height, width = img.shape
    
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
    
    return (depth_min, depth_max,
            height_min, height_max,
            width_min, width_max)

class CenterCrop(Crop):
    """Crop the image 

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
        dim (int) : Dimension of the input volumes (2D or 3D)
    """

    def __init__(self, output_size, dim=3, do_affine=False):
        super(CenterCrop, self).__init__(output_size, dim)

    def __call__(self, sample):
        
        new_sample = []
        for (irm, mask) in sample:
            
            crop_shape = center_crop_indices(irm, self.output_size)

            new_irm = crop(irm, crop_shape)
            new_mask = None if mask is None else crop(mask, crop_shape)
            new_sample.append((new_irm, new_mask))

        return new_sample


class RandomCrop(Crop):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, dim=3):
        super(RandomCrop, self).__init__(output_size, dim)

    def __call__(self, sample):
        
        irm = sample[0][0]
        
        depth = min( [sample[i][0].shape[1] for i in range(len(sample)) ] )
        height = min( [sample[i][0].shape[2] for i in range(len(sample)) ] )
        width = min( [sample[i][0].shape[3] for i in range(len(sample)) ] )

        i = random.randint(0, depth - self.output_size[0])
        j = random.randint(0, height - self.output_size[1])
        k = random.randint(0, width - self.output_size[2])
        
        crop_shape = (i, i + self.output_size[0],
                      j, j + self.output_size[1],
                      k, k + self.output_size[2])
        
        new_sample = []
        for (irm, mask) in sample:
            
            new_irm = crop(irm, crop_shape)
            new_mask = crop(mask, crop_shape)
            new_sample.append((new_irm, new_mask))

        return new_sample

W_list = [400, 800, 1000]
L_list = [40, 100, 400]

def window(W, L, img):
    
    value_min = L - (W/2)
    value_max = L + (W/2)
    
    img = np.clip(img, value_min, value_max)
    
    mini = np.min(img)
    maxi = np.max(img)
    
    array = (img - mini) / (maxi - mini)
    
    return array

def normalize(img, multi_windows=False):
     
    if multi_windows:
        
        arrays_list = []
        for W, L in zip(W_list, L_list):
            
            array = window(W, L, img)

            arrays_list.append(array)
        array = np.concatenate(arrays_list, axis=0)
        
    else:
        W, L = W_list[0], L_list[0]
        
        array = window(W, L, img)

    return array

class Normalize(object):
    """ Normalize the dicom image in sample. The dicom image must be a Tensor"""
    
    def __init__(self, multi_windows=False):
        
        self.multi_windows = multi_windows
        
    def __call__(self, sample):
        
        new_sample = []
        for (irm, mask) in sample:
            
            new_irm = normalize(irm, self.multi_windows)
            new_sample.append((new_irm, mask))

        return new_sample


class AxialFlip(object):

    def __call__(self, sample):

        choice_x = random.randint(0, 1)
        choice_y = random.randint(0, 1)
        choice_z = random.randint(0, 1)
        
        new_sample = []
        
        for (irm, mask) in sample:
            new_irm = self.axialflip(irm, choice_x, choice_y, choice_z)
            new_mask = None if mask is None else self.axialflip(mask, choice_x, choice_y, choice_z)
            new_sample.append((new_irm, new_mask))

        return new_sample

    def axialflip(self, array, choice_x, choice_y, choice_z):

        ndim = len(array.shape)

        if choice_x == 1:
            if ndim == 3:
                array = array[:, :, ::-1]
            elif ndim == 4:
                array = array[:, :, :, ::-1]
            else:
                raise ValueError

        if choice_y == 1:
            if ndim == 3:
                array = array[:, ::-1, :]
            elif ndim == 4:
                array = array[:, :, ::-1, :]
            else:
                raise ValueError
                
        if choice_z == 1:
            if ndim == 3:
                array = array[::-1, ...]
            elif ndim == 4:
                array = array[:, ::-1, :, :]
            else:
                raise ValueError

        return np.ascontiguousarray(array)


class RandomRotation90(object):
    '''
        Taken from augment_rot90 from MIC-DKFZ/batchgenerators
        https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/spatial_transformations.py
    '''

    def __init__(self, num_rot=(1, 2, 3, 4), axes=(0, 1, 2)):

        self.num_rot = num_rot
        self.axes = axes

    def __call__(self, sample):

        num_rot = random.choice(self.num_rot)
        axes = random.sample(self.axes, 2)
        axes.sort()
        axes = [i + 1 for i in axes] # img has shap of lenght 4
        
        def f(img):
            return np.ascontiguousarray(np.rot90(img, num_rot, axes))
        
        new_sample = []
        for irm, mask in sample:
            new_irm = f(irm)
            new_mask = None if mask is None else f(mask)
            new_sample.append((new_irm, new_mask))
        return new_sample

