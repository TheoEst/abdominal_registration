# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:22 2020

@author: t_estienne
"""
import ants
import os
from tqdm import tqdm

home = './data/'
main_path = home + 'Pancreas-CT/resample/'
save_path = home + 'Pancreas-CT/ants_warped/'

end = '.nii.gz'
n = len(end)

reference_path = home + 'L2R_Task3_AbdominalCT/Training/img/img0001.nii.gz'
reference = ants.image_read(reference_path)
reference_array = reference.numpy()
reference_array = reference_array[::-1, :, :]
reference = reference.new_image_like(reference_array)

img_folder =  main_path + '/nifti/'
save_folder = save_path + '/nifti/'

label_folder =  main_path + 'labels/'
label_save_folder = save_path + 'labels/'
    
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
    os.makedirs(label_save_folder)


patients = [f[:-n] for f in os.listdir(img_folder) if f.endswith(end)]

for patient in tqdm(patients):
    
    moving_path = img_folder + patient + end
    seg_path = label_folder + patient + '-seg' + end
    
    try:
        moving = ants.image_read(moving_path)
        segmentation = ants.image_read(seg_path)
        
        
        mytx = ants.registration(fixed=reference , moving=moving, 
                                 type_of_transform='Translation')

        warped_moving = ants.apply_transforms(fixed=reference, moving=moving,
                                                    transformlist=mytx['fwdtransforms'],
                                                    defaultvalue=-1024)
        
        warped_segmentation = ants.apply_transforms(fixed=reference, moving=segmentation,
                                                    transformlist=mytx['fwdtransforms'],
                                                    interpolator='nearestNeighbor',
                                                    defaultvalue=0)
        
        ants.image_write(warped_moving, save_folder + patient.lower() + end)
        ants.image_write(warped_segmentation, label_save_folder + patient.lower() + '-seg' + end)
    
    except Exception as e:
        print('Exception : {}'.format(e))
        print('Patient : {}'.format(patient))
                
            
