# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:49:01 2020

@author: t_estienne
"""
import os
import argparse
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm 
import SimpleITK as sitk
import numpy as np
import skimage

# My package
from abdominal_registration import Dataset
from abdominal_registration import model_loader
from abdominal_registration import utils
from abdominal_registration import transformations
from abdominal_registration import main_seg
from abdominal_registration import numpy2nifty
from abdominal_registration.Dataset import cohort2folder

repo = 'abdominal_registration/'
main_path = './' + repo
model_names = ['SegNet']

def parse_args():

    parser_main = main_seg.parse_args(add_help=False)

    parser = argparse.ArgumentParser(
        description='Keras automatic registration',
        parents=[parser_main])


    parser.add_argument('--train', action='store_true',
                        help='Calcul the output of the train dataset')
    
    parser.add_argument('--val', action='store_true',
                        help='Calcul the output of the val dataset')
    
    return parser

        
def predict(args):

    # Init of args
    args.cuda = torch.cuda.is_available()
    args.data_parallel = args.data_parallel and args.cuda
    args.merge_train_val, args.test = False, False
    args.inference = True

    print('CUDA available : {}'.format(args.cuda))
    
    args.val_cohorts = args.cohorts if len(args.val_cohorts) == 0 else args.val_cohorts
    if isinstance(args.crop_size, int):
        args.crop_size = (args.crop_size, args.crop_size, args.crop_size)

    if args.channels is None:
        args.channels = [4, 8, 16, 32, 64, 128, 256]

    if args.classic_vnet:
        args.nb_Convs = [1, 2, 3, 2, 2, 2]
    elif args.nb_Convs is None:
        args.nb_Convs = [1, 1, 1, 1, 1, 1, 1]
        
    
    args.gpu = 0
    
    args.main_path = main_path
    args.save_path = args.main_path + 'save/'
    args.model_path = args.save_path + 'models/'    
    args.dataset_path = args.main_path + '/datasets/'
    args.pseudo_seg_path = args.main_path + 'data/'
    
    # Create folders if don't exist
    folders = []
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
    # Model
    print('Load model ...')
    model_kwargs = {}
    args.arch = 'SegNet'
    params = ['channel_multiplication', 'pool_blocks', 'channels',
              'instance_norm', 'batch_norm', 'nb_Convs']
    
    for param in params:
        model_kwargs[param] = getattr(args, param)

    (model, 
     model_epoch) = model_loader.load_model(args, model_kwargs)
    
    if args.model_abspath is not None:
        model_name = args.model_abspath.split('/')[-2]
    else:
        model_name, model_epoch = model_epoch.split('::')
        
    print('=> Model ready')
    print(model)
    
    if args.data_parallel:
        model = nn.DataParallel(model).cuda(args.gpu)

    elif args.cuda:
        model = model.cuda(args.gpu)
    
    model.eval()

    # Data        
    crop = transformations.CenterCrop(args.crop_size)
    transforms_list = [transformations.Normalize(), crop]
    transformation = torchvision.transforms.Compose(transforms_list)
    
    
    (train_Dataset, 
     val_Dataset) = Dataset.init_datasets(transformation, transformation,
                                          args, segmentation=True)
    loader_kwargs = {'batch_size':args.batch_size, 'shuffle' : False,
                     'num_workers':args.workers, 'pin_memory' : False,
                     'drop_last' : False}
    
    if args.train:
        train_loader = torch.utils.data.DataLoader(train_Dataset, 
                                                   **loader_kwargs)
    if args.val:
        val_loader = torch.utils.data.DataLoader(val_Dataset, 
                                                 **loader_kwargs)

    with torch.no_grad():
        
        if args.train:
            inference(train_loader, model, args)
            
        if args.val:
            inference(val_loader, model, args)


def keep_connected_components(mask):
    
    channels = np.max(mask)
    
    for channel in range(1, channels +1):
        mask_ = mask == channel
        labels, num = skimage.measure.label(mask_, background=0, 
                                            return_num=True)
        if num>0:
            volume = [np.sum(labels == i) for i in range(1, num+1)]
        
            sorted_volume = np.argsort(volume)
            biggest_label = sorted_volume[-1] + 1
        
            mask[labels == biggest_label] = channel
            mask[~np.isin(labels, [biggest_label,0])] = 0
        
    return mask
        
def save_seg(mask_pred, patient, cohort, args):

    mask_pred = utils.to_numpy(args, mask_pred).squeeze()
    
    mask_pred = np.argmax(mask_pred, axis=0)
    #mask_pred = keep_connected_components(mask_pred)
    mask_pred = mask_pred[::-1, ::-1, ::-1]
        
    if cohort in cohort2folder.keys():
        data_path = args.pseudo_seg_path  + 'Medical_Decathlon/ants_warped/' + cohort2folder[cohort]
        pred_path = data_path + 'pseudo_seg/'
        data_path += 'imagesTr/'
        
    elif cohort == 'tcia_pancreas':
        data_path = args.pseudo_seg_path + 'Pancreas-CT/ants_warped/'
        pred_path = data_path + 'pseudo_seg/'
        data_path += 'nifti/'
        
    elif cohort == 'kits':
        data_path = args.pseudo_seg_path + 'kits19/ants_warped/' + patient + '/'
        pred_path = data_path

    else:
        data_path = args.pseudo_seg_path + 'L2R_Task3_AbdominalCT/Training/'
        pred_path = data_path + 'pseudo_seg/'
        data_path += 'img/'
    

    if not os.path.isdir(pred_path):
        os.makedirs(pred_path)
        
    if cohort == 'kits':
        sitk_img = sitk.ReadImage(data_path + 'imaging.nii.gz')
    else:
        sitk_img = sitk.ReadImage(data_path + patient + '.nii.gz')
   
    sitk_mask = numpy2nifty.numpy2nifty(mask_pred, sitk_img, args)
    
    patient = patient.split('/')[-1]
    
    if cohort == 'kits':
        path = pred_path + 'pseudo_seg-seg.nii.gz'
    else:
        path = pred_path + patient + '-seg.nii.gz'
    
    sitk.WriteImage(sitk_mask, path)
    
def inference(loader, model, args):
        
    for i, gt_sample in tqdm(enumerate(loader, 1)):
        
        patients, cohort = gt_sample['patient'], gt_sample['cohort']
        
        ct = gt_sample['ct']
        ct = utils.to_var(args, ct.float())
        label, gt_mask = gt_sample['label'], gt_sample['mask']
        label = utils.to_numpy(args, label)
        
        # compute output
        mask_pred = model(ct)
        
        # Keep the groundtruth labels
        # Loop on all patient and all labels
        for i in range(ct.shape[0]):
            for j in range(1, label.shape[1]):
                if label[i, j] == 1:
                    mask_pred[i, j, ...] = gt_mask[i, j, ...]
        
        n = ct.shape[0]
        for batch in range(n):
            save_seg(mask_pred[batch, ...], patients[batch], 
                     cohort[batch], args)


if __name__ == '__main__':

    parser = parse_args()
    args = parser.parse_args()

    predict(args)
