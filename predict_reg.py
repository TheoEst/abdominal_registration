# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:15:10 2020

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

# My package
from abdominal_registration import Dataset
from abdominal_registration import model_loader
from abdominal_registration import utils
from abdominal_registration import transformations
from abdominal_registration import main
from abdominal_registration import FrontiersNet

repo = 'abdominal_registration/'
main_path = './' + repo

def parse_args():

    parser_main = main.parse_args(add_help=False)

    parser = argparse.ArgumentParser(
        description='Keras automatic registration',
        parents=[parser_main])


    parser.add_argument('--train', action='store_true',
                        help='Calcul the output of the train dataset')
    
    parser.add_argument('--val', action='store_true',
                        help='Calcul the output of the val dataset')

    parser.add_argument('--save-grid', action='store_true', default=False, 
                        help='Store the mask when predicting output of network')
    
    parser.add_argument('--save-grid-numpy', action='store_true', default=False, 
                        help='Store the mask when predicting output of network')
    
    parser.add_argument('--save-submission', action='store_true', default=False, help='Save the submission for leaderboard')
    
    parser.add_argument('--save-deformed-img', action='store_true', default=False,
                        help='Save the deformed img as numpy file')

    return parser


def predict(args):
    
    args.main_path = main_path
    args.val_cohorts = args.cohorts if len(args.val_cohorts) == 0 else args.val_cohorts
    args.add_extra_organs = False

    # Init of args
    args.cuda = torch.cuda.is_available()
    args.data_parallel = args.data_parallel and args.cuda
    print('CUDA available : {}'.format(args.cuda))

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
    
    # Model
    print('Load model ...')
    model_kwargs = {}

    if args.arch in ['FrontiersNet']:
        params = ['channel_multiplication', 'pool_blocks', 'channels',
                  'last_activation', 'instance_norm', 'batch_norm',
                  'activation_type', 'nb_Convs', 'multi_windows',
                  'freeze_registration', 'zeros_init',
                  'symmetric_training', 'deep_supervision']
        
    for param in params:
        model_kwargs[param] = getattr(args, param)

    (model, 
     model_epoch) = model_loader.load_model(args, model_kwargs)
    
    model_name = args.model_abspath.split('/')[-1]
    if '.' in model_name:
        model_name = model_name.split('.')[0]

    args.pred_path = args.save_path + 'pred/' + model_name + '/'
    args.submission_path = args.save_path + 'submission/' + model_name + '/task_03/'


    # Create folders if don't exist
    folders = [args.pred_path, args.submission_path]
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    
    print('=> Model ready')
    print(model)
    
    if args.data_parallel:
        model = nn.DataParallel(model).cuda(args.gpu)
    elif args.cuda:
        model = model.cuda(args.gpu)
    
    model.eval()

    # Data        
    crop = transformations.CenterCrop(args.crop_size)

    transforms_list = [transformations.Normalize(args.multi_windows), crop]
    
    transformation = torchvision.transforms.Compose(transforms_list)
    

    (train_Dataset, 
     val_Dataset) = Dataset.init_datasets(transformation, transformation, args,
                                          registration=True)
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

def numpy2nifty(array, sitk_img):

    img = sitk.GetImageFromArray(array)
    
    img.SetDirection(sitk_img.GetDirection())
    img.SetSpacing(sitk_img.GetSpacing())
    img.SetOrigin(sitk_img.GetOrigin())
    
    return img

def convert_pytorch_grid2scipy(grid):
    '''
        Convert from the pytorch grid_sample formulation to the scipy formulation
    '''
    _, H, W, D = grid.shape
    grid_x  = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2
    
    grid = np.stack([grid_z, grid_y, grid_x])
    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid
    
    # Simple ITK to nibabel grid
    grid = grid[::-1, ...] 
    grid = grid.swapaxes(1, 3)
    
    return grid

def save_pred(deformed_img, deformable_grid, integrated_grid,
              deformed_mask, moving_patient, reference_patient,
              args):

    deformed_img = deformed_img.squeeze()[0, ...]
    deformed_mask = deformed_mask.squeeze()
    deformed_mask = np.argmax(deformed_mask, axis=0)

    data_path = main_path + 'data/L2R_Task3_AbdominalCT/'
    data_path += 'Testing/' if args.test else 'Training/'
    
    irm_path = data_path + 'img/{}.nii.gz'.format(moving_patient)
    
    if args.save_deformed_img:

        sitk_img = sitk.ReadImage(irm_path)
    
        deformed_mask = numpy2nifty(deformed_mask, sitk_img)
        deformed_img = numpy2nifty(deformed_img, sitk_img)
    
        pred_path = args.pred_path + moving_patient + '_' + reference_patient
        path = pred_path + '-deformed_mask-seg.nii.gz'
    
        sitk.WriteImage(deformed_mask, path)
    
        path = pred_path + '-deformed_img.nii.gz'
    
        sitk.WriteImage(deformed_img, path)
    
    integrated_grid = integrated_grid.squeeze()
    
    if args.save_grid:
        deformable_grid =  deformable_grid.squeeze()
        
        for i, x in enumerate(['x', 'y', 'z']):
            
            path = pred_path + '_deformed_grid_' + x + '.nii.gz'
            grid_x = numpy2nifty(deformable_grid[i, ...], sitk_img)
            sitk.WriteImage(grid_x, path)
        
            path = pred_path + '_integrated_grid_' + x + '.nii.gz'
            grid_x = numpy2nifty(integrated_grid[i, ...], sitk_img)
            sitk.WriteImage(grid_x, path)
    
    if args.save_grid_numpy:
        deformable_grid = deformable_grid.squeeze()
        path = pred_path + '_deformed_grid.npy'
        np.save(path, deformable_grid)
        np.save(pred_path + '_integrated_grid.npy', integrated_grid)

    if args.save_submission:
        # Reverse the flip done on the images
        integrated_grid = integrated_grid[:, ::-1, ::-1, :]
        integrated_grid[2, ...] = - integrated_grid[2, ...]
        integrated_grid[1, ...] = - integrated_grid[1, ...]

        scipy_grid = convert_pytorch_grid2scipy(integrated_grid)
        path = args.submission_path + 'disp_{}_{}.npy'.format(
            reference_patient[3:],
            moving_patient[3:])
        np.save(path, scipy_grid)
        
def inference(loader, model, args):
    

    for i, gt_sample in tqdm(enumerate(loader, 1)):
        
        (reference, moving) = (gt_sample['reference_ct'], 
                               gt_sample['moving_ct'])
        
        reference_patients, moving_patients = (gt_sample['reference_patient'], 
                                               gt_sample['moving_patient'])
        reference = utils.to_var(args, reference.float())
        moving = utils.to_var(args, moving.float())
        
        # compute output
        (deformable_grid, integrated_grid, 
        deformed_img) = model(moving, reference)[0]
        
        moving_mask = utils.to_var(args, gt_sample['moving_mask'].float())

        deformed_moving_mask = FrontiersNet.diffeomorphic3D(moving_mask,
                                                            integrated_grid)
        
        deformed_moving_mask = (deformed_moving_mask > 0.5).float()
        
        n = reference.shape[0]
        deformed_img = utils.to_numpy(args, deformed_img)
        deformable_grid = utils.to_numpy(args, deformable_grid)
        integrated_grid = utils.to_numpy(args, integrated_grid)
        deformed_moving_mask = utils.to_numpy(args, deformed_moving_mask)
        
        for batch in range(n):
            save_pred(deformed_img[batch, ...],
                      deformable_grid[batch, ...],
                      integrated_grid[batch, ...],
                      deformed_moving_mask[batch, ...],
                      moving_patients[batch], reference_patients[batch],
                      args)
        
        torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = parse_args()
    args = parser.parse_args()
    
    predict(args)
