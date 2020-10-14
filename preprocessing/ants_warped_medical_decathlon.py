# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:02:21 2020

@author: t_estienne
"""
import ants
import os
from tqdm import tqdm

home = './data/'
main_path = home + 'Medical_Decathlon/resample/'
save_path = home + 'Medical_Decathlon/ants_warped/'

cohorts = ['Task03_Liver/', 'Task07_Pancreas/', 'Task08_HepaticVessel/',
           'Task09_Spleen/', 'Task10_Colon/']

end = '.nii.gz'
n = len(end)


reference_path = home + 'L2R_Task3_AbdominalCT/Training/img/img0001.nii.gz'
reference = ants.image_read(reference_path)
reference_array = reference.numpy()
reference_array = reference_array[::-1, :, :]
reference = reference.new_image_like(reference_array)


for cohort in cohorts:
    print('************** Starting cohort : {} **************'.format(cohort))

    folder =  main_path + cohort + 'imagesTr/'
    save_folder = save_path + cohort + 'imagesTr/'
    
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    
    label_folder =  main_path + cohort + 'labelsTr/'
    label_save_folder = save_path + cohort + 'labelsTr/'
            
    if not os.path.isdir(label_save_folder):
        os.makedirs(label_save_folder)
        
    patients = [f[:-n] for f in os.listdir(folder) if f.endswith(end)]
    
    for patient in tqdm(patients):
        
        moving_path = folder + patient + end
        pseudo_seg_path = label_folder + patient + '-seg' + end
        
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
            
            ants.image_write(warped_moving, save_folder + patient.lower() + end)
            ants.image_write(warped_segmentation, label_save_folder + patient.lower() + '-seg' + end)
        
        except Exception as e:
            print('Exception : {}'.format(e))
            print('Patient : {}'.format(patient))
