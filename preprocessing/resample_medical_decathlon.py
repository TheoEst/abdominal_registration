# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:20:28 2020

@author: T_ESTIENNE
"""
import os 
import SimpleITK as sitk
from tqdm import tqdm 

home = './data/'
data_path = home + 'Medical_Decathlon/'
save_path = home + 'Medical_Decathlon/resample/'
cohorts = ['Task03_Liver/', 'Task07_Pancreas/', 'Task08_HepaticVessel/',
           'Task09_Spleen/', 'Task10_Colon/']
train_test_folders = ['imagesTr/', 'imagesTs/']


end = '.nii.gz'
n = len(end)

def load_sitk(path):
    img = sitk.ReadImage(path)
    return img

def load_patient(patient, folder):
    
    ct = load_sitk(folder + patient + end)

    return ct

def do_resampling(image, spacing_x, spacing_y, spacing_z, interpolator='Linear'):

    if interpolator == 'Linear':
        interpolator = sitk.sitkLinear
    else:
        interpolator = sitk.sitkNearestNeighbor

    # to calcul new dimensions
    spacing = image.GetSpacing()
    size = image.GetSize()
    fact_x = spacing[0] / spacing_x
    fact_y = spacing[1] / spacing_y
    fact_z = spacing[2] / spacing_z
    size_x = int(round(size[0] * fact_x))
    size_y = int(round(size[1] * fact_y))
    size_z = int(round(size[2] * fact_z))
    # to do resampling
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(image)
    f.SetOutputOrigin(image.GetOrigin())
    f.SetOutputSpacing((spacing_x, spacing_y, spacing_z))
    f.SetSize((size_x, size_y, size_z))
    f.SetInterpolator(interpolator)
    f.SetOutputPixelType(sitk.sitkInt16)
    result = f.Execute(image)
    #result_around = sitk.GetImageFromArray(
    #    np.around(sitk.GetArrayFromImage(result)))
    return result

for cohort in cohorts:
    print('************** Starting cohort : {} **************'.format(cohort))
    for train_test in train_test_folders:
        
        folder =  data_path + cohort + train_test
        save_folder = save_path + cohort + train_test
    
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        if train_test == 'imagesTr/':
            label_folder =  data_path + cohort + 'labelsTr/'
            label_save_folder = save_path + cohort + 'labelsTr/'
                    
            if not os.path.isdir(label_save_folder):
                os.makedirs(label_save_folder)
                        
        patients = [f[:-n] for f in os.listdir(folder) if f.endswith(end)]
        
        for patient in tqdm(patients):
        
            ct = load_patient(patient, folder)
            spacing = ct.GetSpacing()
            shape = ct.GetSize()
            ct = do_resampling(image=ct, spacing_x=2, spacing_y=2, spacing_z=2, 
                               interpolator='Linear')
            
            # prefix name
            sitk.WriteImage(ct, save_folder + patient.lower() + end)

            if train_test == 'imagesTr/':
                
                mask = load_patient(patient, label_folder)
                mask = do_resampling(image=mask, spacing_x=2, spacing_y=2, 
                                     spacing_z=2,
                                     interpolator='NearestNeighbor')
                
                # prefix name
                sitk.WriteImage(mask, label_save_folder + patient.lower() + '-seg' + end)
