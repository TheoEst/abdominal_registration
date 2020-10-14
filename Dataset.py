# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:26:48 2020

@author: T_ESTIENNE
"""
import numpy as np
import pandas as pd
import time
import torch.utils.data as data
import SimpleITK as sitk
from tqdm import tqdm 
import random
import nibabel.freesurfer.mghformat as mgh
import os

end = '.nii.gz'
seg_name = '_seg'

cohort2folder = {'liver' : 'Task03_Liver/', 'pancreas' : 'Task07_Pancreas/',
                 'hepatic' : 'Task08_HepaticVessel/', 
                 'spleen' : 'Task09_Spleen/', 'colon' : 'Task10_Colon/'}
    
def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def load_mgh(path):
    return mgh.load(path).get_data()

def load_dataset(dataset_path, cohort, segmentation=False,
                 test=False):
        
        
    train_set = np.loadtxt(dataset_path + cohort + '_train.txt', dtype=str)
    val_or_test = 'test' if (test and cohort == 'learn2reg_task3') else 'val'
    val_set = np.loadtxt(dataset_path + cohort + '_{}.txt'.format(val_or_test), 
                         dtype=str)
    val_pairs = None
    
    if cohort == 'learn2reg_task3':
        val_pairs = pd.read_csv(dataset_path + 'task_03_pairs_{}.csv'.format(val_or_test)).values

    if cohort in cohort2folder.keys():
        train_set = np.concatenate([train_set, val_set], axis=0)
        val_set = []
    
    if cohort in ['tcia_pancreas', 'kits'] and (not segmentation):
        test_set = np.loadtxt(dataset_path + cohort + '_test.txt', dtype=str)
        train_set = np.concatenate([train_set, val_set, test_set], axis=0)
        val_set = []
        
    return train_set, val_set, val_pairs

def convert_tcia_labels(mask, keep_all_label=False):
    """
    1 -> spleen, 3 -> left kidney, 4 -> gallbladder, 5 -> esophagus
    6 -> liver, 7 - > stomach, 11 -> pancreas, 14 -> duodenum
    """
    
    mask[np.isin(mask, [14])] = 0 # Remove duodenum
    label = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1] # no right kidney

    if keep_all_label:
        label += [0,0]

    return mask, label

def convert_L2R_labels(mask, keep_all_label=False):
    
    if keep_all_label:
        label = 14*[1]
        
    else:
        mask[np.isin(mask, [12, 13])] = 0 # Remove adrenal gland
        label = 12*[1]
        
    return mask, label


def convert_kits(mask,  keep_all_label=False):
    
    mask[mask == 2] = 0  # Remove tumor
    x,y, z = np.where(mask == 1)

    mean_value = np.mean(z)
    
    left_index = z < mean_value
    right_index = z > mean_value
    mask[x[left_index], y[left_index], z[left_index]] = 2
    mask[x[right_index], y[right_index], z[right_index]] = 3
    
    label = 12*[0]
    if keep_all_label:
        label += [0,0]

    label[2] = 1
    label[3] = 1
    
    return mask, label

def convert_medical_decathlon_labels(mask, cohort, keep_all_label=False):
    """
    1 spleen, 6 liver, 11 pancreas
    """
    label = 12*[0]
    if keep_all_label:
        label += [0,0]
    
    if cohort == 'liver':
        mask[mask == 2] = 6
        mask[mask == 1] = 6
        label[6] = 1
    
    elif cohort == 'pancreas':
        mask[mask == 2] = 11
        mask[mask == 1] = 11
        label[11] = 1

    elif cohort == 'spleen':
        mask[mask != 1] = 0
        label[1] = 1                      

    elif cohort == 'hepatic':
        mask[mask == 2] = 0
        mask[mask == 1] = 0
        
    return mask, label

class BrainDataset(data.Dataset):
    
    def __init__(self, files_list, folder, transform=None,
                 verbosity=False, pseudo_labels=False, keep_all_label=False):
        
        super(BrainDataset, self).__init__()
        self.files_list = files_list
        self.folder = folder
        self.transform = transform
        self.verbosity = verbosity
        self.pseudo_labels = pseudo_labels
        self.keep_all_label = keep_all_label


    def load(self, patient, cohort='learn2reg_task3'):
        
        if cohort == 'tcia_pancreas':
            return self.load_tcia_pancreas(patient)
        elif cohort == 'learn2reg_task3':
            return self.load_learn2reg(patient)
        elif cohort == 'kits':
            return self.load_kits(patient)
        else:
            return self.load_medical_decathlon(patient, cohort)
        
    
    def load_learn2reg(self, patient):
        """
        1 spleen, 2 right kidney, 3 left kidney, 4 gall bladder, 5 esophagus
        6 liver, 7 stomach, 8 aorta, 9 inferior vena cava, 
        10 portal and splenic vein, 11 pancreas,
        12 left adrenal gland, 13 right adrenal gland.
        """
        start = time.time()
        
        ct_path = self.folder + 'img/' + patient + end
        
        array = load_sitk(ct_path)[np.newaxis, ...].astype(float)# Add 1 dimension
        array = array[:, ::-1, ::-1, :]
        
        if self.load_mask:

            if self.pseudo_labels:
                mask_path = self.folder + 'pseudo_seg/' + patient + '-seg' + end
            else:
                mask_path = self.folder + 'label/label' + patient[3:] + end
            mask = load_sitk(mask_path).astype(int)
            
            if self.pseudo_labels:
                label = 12*[1]
            else:
                mask, label = convert_L2R_labels(mask, self.keep_all_label)

            mask = mask[::-1, ::-1, :]

            num_label = len(label)
            mask = np.rollaxis(np.eye(num_label, dtype=np.uint8)[mask], -1, 0)
        else:

            mask = np.zeros(array.shape[1:], dtype=np.int16)
            mask = np.rollaxis(np.eye(2)[mask], -1, 0) # Put to one hot encoder
            label = 14*[0]
        
        stop = time.time()

        return array, mask, np.array(label), stop - start
    
    def load_medical_decathlon(self, patient, cohort):

        start = time.time()
        
        ct_path = self.folder + 'imagesTr/' + patient  + end
        array = load_sitk(ct_path)[np.newaxis, ...].astype(float)# Add 1 dimension
        
        # Correction of the orientation
        array = array[:, ::-1, ::-1, ::-1]
        
        if self.load_mask:

            if self.pseudo_labels:
                mask_path = self.folder + 'pseudo_seg/' + patient + '-seg' + end
            else:
                mask_path = self.folder + 'labelsTr/' + patient + '-seg' + end
                
            mask = load_sitk(mask_path).astype(int)
            mask = mask[::-1, ::-1, ::-1]
            
            if self.pseudo_labels:
                label = 12*[1]
            else:
                mask, label = convert_medical_decathlon_labels(mask, cohort,
                                                               self.keep_all_label)

            num_label = len(label)
            mask = np.rollaxis(np.eye(num_label, dtype=np.uint8)[mask], -1, 0)
        else:
            mask = np.zeros(array.shape[1:], dtype=np.int16)
            mask = np.rollaxis(np.eye(2)[mask], -1, 0)# Put to one hot encoder
            label = 14*[0]
        
        stop = time.time()

        return array, mask, np.array(label), stop - start
        
    def load_tcia_pancreas(self, patient):
        """
        1 -> spleen, 3 -> left kidney, 4 -> gallbladder, 5 -> esophagus
        6 -> liver, 7 - > stomach, 11 -> pancreas, 14 -> duodenum
        """
        start = time.time()
        
        ct_path = self.folder + 'nifti/' + patient  + end
        array = load_sitk(ct_path)[np.newaxis, ...].astype(float)# Add 1 dimension
        
        # Correction of the orientation
        array = array[:, ::-1, ::-1, ::-1]
        
        if self.pseudo_labels:
            mask_path = self.folder + 'pseudo_seg/' + patient + '-seg' + end
        else:
            mask_path = self.folder + 'labels/' + patient + '-seg' + end

        if self.load_mask and os.path.isfile(mask_path):
            
            mask = load_sitk(mask_path).astype(int)
            mask = mask[::-1, ::-1, ::-1]

            if self.pseudo_labels:
                label = 12*[1]
            else:
                mask, label = convert_tcia_labels(mask, self.keep_all_label)
                
            num_label = len(label)
            mask = np.rollaxis(np.eye(num_label, dtype=np.uint8)[mask], -1, 0)
        else:
            print(mask_path)
            mask = np.zeros(array.shape[1:], dtype=np.int16)
            mask = np.rollaxis(np.eye(2)[mask], -1, 0) # Put to one hot encoder
            label = 14*[0]
        
        stop = time.time()
        

        return array, mask, np.array(label), stop - start
    
    def load_kits(self, patient):
        """
        1 - kidney (left and right);  2 - tumor
        """
        start = time.time()
        ct_path = self.folder + patient  + '/imaging' + end
        array = load_sitk(ct_path)[np.newaxis, ...].astype(float)# Add 1 dimension
        
        # Corection of orientation
        array = array[:, ::-1, ::-1, ::-1]
        
        if self.pseudo_labels:
            mask_path = self.folder + patient + '/pseudo_seg-seg' + end
        else:
            mask_path = self.folder + patient + '/segmentation-seg' + end
                
        if self.load_mask  and os.path.isfile(mask_path):
                            
            mask = load_sitk(mask_path).astype(int)
            mask = mask[::-1, ::-1, ::-1]

            if self.pseudo_labels:
                label = 12*[1]
            else:
                mask, label = convert_kits(mask, self.keep_all_label)
                
            num_label = len(label)
            mask = np.rollaxis(np.eye(num_label, dtype=np.uint8)[mask], -1, 0)
        else:
            print(mask_path)
            mask = np.zeros(array.shape[1:], dtype=np.int16)
            mask = np.rollaxis(np.eye(2)[mask], -1, 0) # Put to one hot encoder
            label = 14*[0]

        stop = time.time()
        
        return array, mask, np.array(label), stop - start
    
class RegistrationDataset(BrainDataset):
    """Dicom dataset."""

    def __init__(self, files_list, folder, transform=None,
                 verbosity=False, cohort='learn2reg_task3',
                 validation=False, pairs=None,
                 pseudo_labels=False, keep_all_label=False,
                 test=False
                 ):

        super(RegistrationDataset, self).__init__(files_list, folder, transform,
                                                  verbosity,
                                                  pseudo_labels=pseudo_labels, 
                                                  keep_all_label=keep_all_label
                                                  )
        
        self.cohort = cohort
        self.validation = validation
        self.load_mask = not test
        self.pairs = pairs
        
    def __len__(self):
        if self.pairs is not None:
            return len(self.pairs)
        else:
            return len(self.files_list)

    def __getitem__(self, idx):
        
        if self.pairs is not None:
            reference_patient, moving_patient = self.pairs[idx, :]
            moving_patient = 'img{:04d}'.format(int(moving_patient)) 
            reference_patient = 'img{:04d}'.format(int(reference_patient))
        else:
            moving_patient = self.files_list[idx]
            reference_patient = random.choice(self.files_list)
        
        start = time.time()
        moving_ct, moving_mask, _, time_load = self.load(moving_patient, self.cohort)
        reference_ct, reference_mask, _,  time_load2 = self.load(reference_patient, self.cohort)
        time_load += time_load2


        sample = [(reference_ct, reference_mask), 
                  (moving_ct, moving_mask)]
        
        if self.transform:
            start_transform = time.time()
            [(reference_ct, reference_mask), 
             (moving_ct, moving_mask)] = self.transform(sample)
            time_transform = time.time() - start_transform
        

        if self.verbosity:
            print('Sample import = {}'.format(time_load))
            print('Sample Transformation = {}'.format(time_transform))
            print('Total time for sample = {}'.format(time.time() - start))
            

        sample = {'reference_ct': reference_ct, 'reference_mask': reference_mask,
                  'moving_ct': moving_ct, 'moving_mask': moving_mask,
                  'reference_patient': reference_patient, 
                  'moving_patient': moving_patient}
        
        return sample


class SegmentationDataset(BrainDataset):
    """Dicom dataset."""

    def __init__(self, files_list, folder, transform=None, verbosity=False,
                 validation=False, cohort='learn2reg_task3',
                 pairs=None, inference=False):

        super(SegmentationDataset, self).__init__(files_list, folder, 
                                                  transform, verbosity)

        self.cohort = cohort
        self.load_mask = not inference

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):

        patient = self.files_list[idx]

        start = time.time()
        ct, mask, label, time_load = self.load(patient, self.cohort)
        sample = [(ct, mask)]
        
        if self.transform:
            start_transform = time.time()
            [(ct, mask)] = self.transform(sample)
            time_transform = time.time() - start_transform
      
        if self.verbosity:
            print('Sample import = {}'.format(time_load))
            print('Sample Transformation = {}'.format(time_transform))
            print('Total time for sample = {}'.format(time.time() - start))
            
        sample = {'ct': ct, 'mask': mask, 'patient': patient,
                  'label': label, 'cohort' : self.cohort}
        
        return sample

      
def init_datasets(transformation, val_transformation, args,
                  segmentation=False, registration=False):
    
    params = ['folder']
            
    if registration:
        Dataset = RegistrationDataset
        params += ['pseudo_labels', 'test', 'keep_all_label']
    elif segmentation:
        Dataset = SegmentationDataset
        params.append('inference')
        segmentation = False if args.inference else segmentation
    
    train_datasets, val_datasets = [], []
    
    data_path = args.main_path + 'data/'
    
    # Load wrong patients
    wrong_patient = np.loadtxt(args.dataset_path + 'wrong_patients.txt',
                               dtype=str)
    
    for cohort in args.cohorts:
        if cohort in cohort2folder.keys():
            args.folder = data_path + 'Medical_Decathlon/ants_warped/'
            args.folder += cohort2folder[cohort]
        elif cohort == 'tcia_pancreas':
            args.folder = data_path + 'Pancreas-CT/ants_warped/'
        elif cohort == 'kits':
            args.folder = data_path + 'kits19/ants_warped/'
        else:
            args.folder = data_path + 'L2R_Task3_AbdominalCT/'
            args.folder += 'Testing/' if args.test else 'Training/'
        
        files_train, files_val, pairs = load_dataset(args.dataset_path, 
                                                     cohort,
                                                     segmentation=segmentation,
                                                     test=args.test)
        
        # Delete wrong patients from files list
        files_train = [f for f in files_train if f not in wrong_patient]
        files_val = [f for f in files_val if f not in wrong_patient]

        if args.merge_train_val:
            files_train = np.concatenate([files_train, files_val])
        
        if args.debug:
            files_train = files_train[:2*args.batch_size+1]
            files_val = files_val[:2*args.batch_size+1]
        
        dataset_kwargs = {'verbosity': args.verbosity,
                          'cohort' : cohort
                          }
        
        for param in params:
            dataset_kwargs[param] = getattr(args, param)

        if cohort in args.val_cohorts:
            val_datasets.append(Dataset(files_val, transform=val_transformation,
                                        validation=True, **dataset_kwargs,
                                        pairs=pairs))
        
        train_datasets.append(Dataset(files_train, 
                                      transform=transformation,
                                      **dataset_kwargs))
    
    train_Dataset = data.ConcatDataset(train_datasets)
    val_Dataset = data.ConcatDataset(val_datasets)

    return train_Dataset, val_Dataset
