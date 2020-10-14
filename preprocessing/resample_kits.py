# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:46:28 2020

@author: t_estienne
"""
import os 
import SimpleITK as sitk
from tqdm import tqdm 

home = './data/'
main_path = home + 'kits19/data/'
save_path = home + 'kits19/resample/'

end = '.nii.gz'
n = len(end)

def load_sitk(path):
    img = sitk.ReadImage(path)
    return img

def load_patient(patient, folder, is_label=False):
    
    if is_label:
        path = folder + patient + '/segmentation' + end
    else:
        path = folder + patient + '/imaging' + end
        
    ct = load_sitk(path)

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

    return result


patients = [f for f in os.listdir(main_path) if os.path.isdir(main_path + f)]

for patient in tqdm(patients):
    

    ct = load_patient(patient, main_path)
    spacing = ct.GetSpacing()
    shape = ct.GetSize()
    ct = do_resampling(image=ct, spacing_x=2, spacing_y=2, spacing_z=2, 
                       interpolator='Linear')

    
    os.makedirs(save_path + patient.lower() + '/')
    # prefix name
    sitk.WriteImage(ct, save_path + patient.lower() + '/imaging' + end)
    
    try:
        mask = load_patient(patient, main_path, is_label=True)
        mask = do_resampling(image=mask, spacing_x=2, spacing_y=2, 
                            spacing_z=2,
                            interpolator='NearestNeighbor')
    
        # prefix name
        sitk.WriteImage(mask, 
                        save_path + patient.lower() + '/segmentation-seg' + end)
    except Exception as e:
        print('Patient {} has no labels'.format(patient))
        print('{}\n'.format(e))
