# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:28 2020

@author: t_estienne
"""
import ants
import os
from tqdm import tqdm

home = './data/'
main_path = home + 'kits19/resample/'
save_path = home + 'kits19/ants_warped/'

end = '.nii.gz'
n = len(end)

reference_path = home + 'L2R_Task3_AbdominalCT/Training/img/img0001.nii.gz'
reference = ants.image_read(reference_path)
reference_array = reference.numpy()
reference_array = reference_array[::-1, :, :]
reference = reference.new_image_like(reference_array)


if not os.path.isdir(save_path):
    os.makedirs(save_path)
  

patients = [f for f in os.listdir(main_path) if os.path.isdir(main_path + f)]

for patient in tqdm(patients):
    
    moving_path = main_path + patient + '/imaging' + end
    pseudo_seg_path = main_path + patient + '/segmentation-seg' + end
            
    try:
        moving = ants.image_read(moving_path)
        segmentation = ants.image_read(pseudo_seg_path)
        
        
        mytx = ants.registration(fixed=reference , moving=moving, 
                                 type_of_transform='Translation')
        
        
        warped_moving = ants.apply_transforms(fixed=reference, moving=moving,
                                                    transformlist=mytx['fwdtransforms'],
                                                    defaultvalue=-1024)
        
        warped_segmentation = ants.apply_transforms(fixed=reference, moving=segmentation,
                                                    transformlist=mytx['fwdtransforms'],
                                                    interpolator='nearestNeighbor',
                                                    defaultvalue=0)
        
        os.makedirs(save_path + patient.lower() + '/')
        ants.image_write(warped_moving, save_path + patient + '/imaging' + end)
        ants.image_write(warped_segmentation, save_path + patient + '/segmentation-seg' + end)
    
    except Exception as e:
        print('Exception : {}'.format(e))
        print('Patient : {}'.format(patient))
