# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:17:13 2020

@author: T_ESTIENNE
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data
from subprocess import call
import sys
import torch.utils.tensorboard as tensorboard
import torchvision
import pandas as pd

# My package
from abdominal_registration import ImageTensorboard
from abdominal_registration import log
from abdominal_registration import transformations
from abdominal_registration import utils
from abdominal_registration import losses
from abdominal_registration import model_loader
from abdominal_registration import Dataset
from abdominal_registration import FrontiersNet

repo = 'abdominal_registration/'
main_path = './' + repo
model_names = ['FrontiersNet']

cohorts = ['liver', 'pancreas', 'spleen', 'colon', 
           'learn2reg_task3', 'tcia_pancreas', 'kits', 'hepatic']


def parse_args(add_help=True):

    parser = argparse.ArgumentParser(
        description='Pytorch automatic registration',
        add_help=add_help)

    parser.add_argument('--arch', '-a', metavar='ARCH', default='FrontiersNet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: VNet)')
    parser.add_argument('--session-name', type=str, default='',
                        help='Give a name to the session')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--batch-size', '-b', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--val-batch-size', default=0, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)')
    parser.add_argument('--print-frequency', '--f', default=1, type=int,
                        metavar='F', help='Print Frequency of the batch (default: 5)')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_false',
                        help='use tensorboard_logger to save data')
    parser.add_argument('--verbosity', action='store_true', 
                        help='Print DataLoader time calculation')
    parser.add_argument('--workers', '-w', default=4, type=int,
                        help='Use multiprocessing for dataloader')
    parser.add_argument('--data-parallel', action='store_false',
                        help='Use data parallel in CUDA')
    parser.add_argument('--save', '-s', action='store_false',
                        help='Save the model during training')
    parser.add_argument('--save-frequency', type=int, default=20,
                        help='Save the model every X epoch')
    parser.add_argument('--image-tensorboard-frequency', type=int, default=5,
                        help='Plot the model in tensorboard every X epoch')
    parser.add_argument('--channel-multiplication', type=int, default=4,
                        help='Divide the number of channels of each convolution')
    parser.add_argument('--pool-blocks', type=int, default=4,
                        help='Number of pooling block (Minimum 2)')
    parser.add_argument('--channels', type=int, default=None, nargs='+',
                        help='List of the channels')
    parser.add_argument('--mse-loss-weight', type=float, default=1,
                        help='Weigths for the reconstruction loss')
    parser.add_argument('--regu-deformable-loss-weight', type=float, default=1,
                        help='Weigths for the defomable regularisation loss')
    parser.add_argument('--deformed-mask-loss-weight', type=float, default=1.,
                        help="Weight for the dice of the deformed mask")
    parser.add_argument('--random-crop', action='store_true',
                        help='Do random crop')
    parser.add_argument('--data-augmentation', action='store_true',
                        help='Add axial flip and data augmentation')
    parser.add_argument('--crop-size', type=int, nargs='+', default=64,
                        help='Crop size')
    parser.add_argument('--val-crop-size', type=int, nargs='+', default=None,
                        help='Crop size')
    parser.add_argument('--last-activation', type=str, default='relu',
                        help='last activation of the decoder')
    parser.add_argument('--activation-type', type=str, default='relu',
                        help='Activation type of the neural network')
    parser.add_argument('--instance-norm', action='store_true',
                        help='Use instance norm during training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='Use batch norm during training')
    parser.add_argument('--nb-Convs', type=int, default=None, nargs='+',
                        help='List of the channels')
    parser.add_argument('--classic-vnet', action='store_true', default=None,
                        help='Use classic')
    parser.add_argument('--freeze-registration', action='store_true', 
                        default=False, help='Freeze the last layer of the registration decoder')
    parser.add_argument('--zeros-init', action='store_true', default=False,
                        help='Initialisation of the last conv of registration')
    parser.add_argument('--deformed-mask-loss', action='store_true', default=False, 
                        help='Add a loss with the dice on deformed mask')
    parser.add_argument('--symmetric-training', action='store_true', default=False,
                        help='Use symmetric training')         
    parser.add_argument('--plot-loss', action='store_true', default=False,
                        help='Plot the registration loss')
    parser.add_argument('--plot-mask', action='store_true', default=False,
                        help='Plot the deformed mask')
    parser.add_argument('--model-abspath', type=str, default=None,
                        help='Absolute path of model to load')
    parser.add_argument('--cohorts', type=str, nargs='+', choices=cohorts,
                        default=['learn2reg_task3'], help='Cohort to use')
    parser.add_argument('--val-cohorts', type=str, nargs='+', choices=cohorts,
                        default=[], help='Cohort to use')
    parser.add_argument('--multi-windows', action='store_true', default=False,
                        help='Use bone and fat windows as extra modality')
    parser.add_argument('--pseudo-labels', action='store_true', default=False,
                        help='Use pseudo labels instead of real labels')
    parser.add_argument('--keep-all-label', action='store_true', default=False, help='Keep all the labels of the L2R abdo datasets')
    parser.add_argument('--deep-supervision', action='store_true', default=False,
                        help='Use deep supervision')
    parser.add_argument('--merge-train-val', action='store_true', default=False,
                        help='Merge train and val for final predictions')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Use test instead of val')
    return parser


def main(args):
    
    
    args.main_path = main_path
    args.val_cohorts = args.cohorts if len(args.val_cohorts)==0 else args.val_cohorts
    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size

    # Init of args
    args.cuda = torch.cuda.is_available()
    args.data_parallel = args.data_parallel and args.cuda

    print('CUDA available : {}'.format(args.cuda))

    if isinstance(args.crop_size, int):
        args.crop_size = (args.crop_size, args.crop_size, args.crop_size)
    val_crop_size = args.val_crop_size if args.val_crop_size is not None else args.crop_size
    
    if args.channels is None:
        args.channels = [4, 8, 16, 32, 64, 128, 256]

    if args.classic_vnet:
        args.nb_Convs = [1, 2, 3, 2, 2, 2]
    elif args.nb_Convs is None:
        args.nb_Convs = [1, 1, 1, 1, 1, 1, 1]
        
    args.gpu = 0

    if args.session_name == '':
        args.session_name = args.arch + '_' + time.strftime('%m.%d %Hh%M')
    else:
        args.session_name += '_' + time.strftime('%m.%d %Hh%M')
    
    if args.debug:
        args.session_name += '_debug'
        
    args.save_path = main_path + 'save/'
    args.model_path = args.save_path + 'models/' + args.session_name
    args.dataset_path = main_path + '/datasets/'
    tensorboard_folder = args.save_path + 'tensorboard_logs/'
    log_folder = args.save_path + 'logs/'

    folders = [args.save_path, args.model_path,
               tensorboard_folder, log_folder, args.dataset_path]
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    if args.tensorboard:
        log_dir = tensorboard_folder + args.session_name + '/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = tensorboard.SummaryWriter(log_dir)
    else:
        writer = None

    print('******************* Start training *******************')
    print('******** Parameter ********')

    # Log
    log_path = log_folder + args.session_name + '.log'
    logging = log.set_logger(log_path)

    # logs some path info and arguments
    logging.info('Original command line: {}'.format(' '.join(sys.argv)))
    logging.info('Arguments:')
    
    for arg, value in vars(args).items():
        logging.info("%s: %r", arg, value)


    # Model
    logging.info("=> creating model '{}'".format(args.arch))

    model_kwargs = {}

    if args.arch in ['FrontiersNet']:
        params = ['channel_multiplication', 'pool_blocks', 'channels',
                  'last_activation', 'instance_norm', 'batch_norm',
                  'activation_type', 'nb_Convs', 'multi_windows',
                  'freeze_registration', 'zeros_init',
                  'symmetric_training', 'deep_supervision']

        for param in params:
            model_kwargs[param] = getattr(args, param)
    
    if args.model_abspath is not None:
        
        (model, 
         model_epoch) = model_loader.load_model(args, model_kwargs)
    else:
        model = model_loader.create_model(args, model_kwargs)
    
    if args.data_parallel:
        model = nn.DataParallel(model).cuda(args.gpu)
    elif args.cuda:
        model = model.cuda(args.gpu)
        
    logging.info('=> Model ready')
    logging.info(model)
    
            
    # Loss
    criterion = {'L1': nn.L1Loss(),
                 'MSE': nn.MSELoss(reduction='none'),
                 'seg': losses.mean_dice_loss,
                 }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data
    if args.random_crop:
        crop = transformations.RandomCrop(args.crop_size)
    else:
        crop = transformations.CenterCrop(args.crop_size)
                
    val_crop = transformations.CenterCrop(val_crop_size)

    transforms_list = [transformations.Normalize(args.multi_windows),
                       crop]
    
    if args.data_augmentation:
        transforms_list.append(transformations.AxialFlip())
        transforms_list.append(transformations.RandomRotation90())
        
    val_transforms_list = [transformations.Normalize(args.multi_windows),
                           val_crop]


    transformation = torchvision.transforms.Compose(transforms_list)
    val_transformation = torchvision.transforms.Compose(val_transforms_list)

    (train_Dataset, 
     val_Dataset) = Dataset.init_datasets(transformation,
                                          val_transformation, args,
                                          registration=True)
                                          
                                          
    train_loader = torch.utils.data.DataLoader(train_Dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=False,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_Dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False,
                                             drop_last=True)
    
    args.identity_grid = utils.to_var(args, 0.5 * torch.ones([args.batch_size,
                                                              3,
                                                              *args.crop_size]))
    
    args.identity_val_grid = utils.to_var(args, 0.5 * torch.ones([val_batch_size,
                                                              3,
                                                              *val_crop_size]))
    
    start_training = time.time()
    best_loss = 1e9

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print('******** Epoch [{}/{}]  ********'.format(epoch+1, args.epochs))
        print(args.session_name)
        start_epoch = time.time()
        
        # train for one epoch
        model.train()
        _ = train(train_loader, model, criterion, optimizer, writer,
                  logging, epoch, args)

        # evaluate on validation set
        with torch.no_grad():
            model.eval()
            avg_loss = train(val_loader, model, criterion,
                                optimizer, writer, logging, epoch,
                                args)
        

        # remember best loss and save checkpoint
        is_best = best_loss > avg_loss
        best_loss = min(best_loss, avg_loss)
        
        utils.save_checkpoint(args,
                              {
                                  'epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'val_loss': avg_loss,
                                  'optizmizer': optimizer.state_dict(),
                              }, is_best)
        
        epoch_time = time.time() - start_epoch
        logging.info('Epoch time : {} s'.format(epoch_time))
        remaining_time_hour = epoch_time * (args.epochs - epoch) // 3600
        remaining_time_min = (epoch_time * (args.epochs - epoch) - (remaining_time_hour*3600)) // 60
        logging.info('Remaining time : {}h {}min'.format(remaining_time_hour, remaining_time_min))

    args.training_time = time.time() - start_training
    logging.info('Finished Training')
    logging.info('Training time : {}'.format(args.training_time))
    log.clear_logger(logging)

    if args.tensorboard:
        writer.close()

    return avg_loss

def calculate_loss(gt_sample, pred_sample, loss_dict,
                   criterion, identity_grid, args, deformed_moving_mask=None,
                   deformed_reference_mask=None):

    loss = 0
    
    L1, MSE, Dice = (criterion['L1'], criterion['MSE'], 
                         criterion['seg'])


    reference = utils.to_var(args, gt_sample['reference_ct'].float())
    moving = utils.to_var(args, gt_sample['moving_ct'].float())
    
    mse_loss, regu_deformable_loss = 0, 0
    
    # 2 : Moving -> Reference, 3 : Reference -> Moving 
    ground_truths = [reference]

    if args.symmetric_training:
        ground_truths += [moving]
        
    index = range(0, len(ground_truths))
    
    for gt, i in zip(ground_truths, index):
        (deformed_img, 
        deformable_grid) = (pred_sample[i][2], pred_sample[i][0])
        
        if args.deep_supervision:
            mse = [MSE(img, gt) for img in deformed_img]
            mse_loss += torch.mean(torch.stack(mse))
        else:
            mse_loss += MSE(deformed_img, gt)
            
        regu_deformable_loss += L1(deformable_grid, identity_grid)
        
    # Reconstruction loss
    mse_loss /= len(index)
    loss_dict.update('MSE_loss', mse_loss.mean().item())
    loss += args.mse_loss_weight * mse_loss.mean()
    
    # Regularisation loss
    regu_deformable_loss /= len(index)
    loss_dict.update('Regu_Loss', regu_deformable_loss.mean().item())
    loss += args.regu_deformable_loss_weight * regu_deformable_loss.mean()
    
    if args.deformed_mask_loss:
        
        reference_mask_gt = utils.to_var(args, gt_sample['reference_mask'].float())
        deformed_mask_loss = Dice(deformed_moving_mask, reference_mask_gt)
        
        if args.symmetric_training:
            
            moving_mask_gt = utils.to_var(args, gt_sample['moving_mask'].float())
            deformed_mask_loss += Dice(deformed_reference_mask, moving_mask_gt)
            deformed_mask_loss /= 2
        
        loss_dict.update('Deformed_mask_Loss', deformed_mask_loss.item())
        loss += args.deformed_mask_loss_weight*deformed_mask_loss


    loss_dict.update('Loss', loss.item())

    return loss, loss_dict, mse_loss

def train(loader, model, criterion, optimizer, writer,
          logging, epoch, args):


    end = time.time()
    loss_dict = utils.MultiAverageMeter(names=[])

    logging_mode = 'Train' if model.training else 'Val'
    columns = ['Spleen', 'Right_Kidney', 'Left_Kidney', 'Gall_Bladder',
               'Esophagus', 'Liver', 'Stomach', 'Aorta', 
               'Inferior_Vena_Cava', 'Portal&Splenic_Vein', 
               'Pancreas']
    
    if args.keep_all_label:
        columns +=  ['Left_Adrenal_Gland', 'Right_Adrenal_Gland']

    dice_dataframe = pd.DataFrame(columns=columns)
    
    for i, gt_sample in enumerate(loader, 1):
        (reference, moving) = (gt_sample['reference_ct'], 
                               gt_sample['moving_ct'])
        
        (reference_patient, moving_patient) = (gt_sample['reference_patient'],
                                               gt_sample['moving_patient'])
        # measure data loading time
        data_time = time.time() - end

        reference = utils.to_var(args, reference.float())
        moving = utils.to_var(args, moving.float())
        moving_mask = utils.to_var(args, gt_sample['moving_mask'].float())
        reference_mask = utils.to_var(args, gt_sample['reference_mask'].float())

        # compute output
        pred_sample = model(moving, reference)

        # Apply predict deformation on ground truth mask
        integrated_grid = pred_sample[0][1]
        deformed_moving_mask = FrontiersNet.diffeomorphic3D(moving_mask, 
                                                            integrated_grid)
        deformed_reference_mask = None
        if args.symmetric_training:
            integrated_grid = pred_sample[1][1]
            deformed_reference_mask = FrontiersNet.diffeomorphic3D(reference_mask, 
                                                                   integrated_grid)
                
            
        # compute loss
        identity_grid = args.identity_grid if model.training else args.identity_val_grid
        loss, loss_dict, mse_loss = calculate_loss(gt_sample, pred_sample, 
                                                   loss_dict, criterion, 
                                                   identity_grid, args,
                                                   deformed_moving_mask, 
                                                   deformed_reference_mask)
        if model.training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        deformed_moving_mask = (deformed_moving_mask > 0.5).float()
        
        if args.symmetric_training:
            deformed_reference_mask = (deformed_reference_mask > 0.5).float()
        
        # Metrics        
        if not model.training:
            patients = gt_sample['moving_patient']
            
            dataframe = losses.evalAllSample(deformed_moving_mask, 
                                             reference_mask, 
                                             patients, args)
            
            dice_dataframe = pd.concat([dice_dataframe, dataframe])
            mean_dice = 0
            for dices in columns:
                loss_dict.update(dices, dataframe.mean()[dices])
                mean_dice += dataframe.mean()[dices]
            loss_dict.update('Mean_Dice', mean_dice / len(columns))
                
        # measure elapsed time
        batch_time = time.time() - end

        end = time.time()  
        
        loss_dict.update('Batch_time', batch_time)
        loss_dict.update('Data_time', data_time)

        if i % args.print_frequency == 0:
            utils.print_summary(epoch, i, len(loader), loss_dict,
                                logging, logging_mode)
        if args.tensorboard:
            step = epoch*len(loader) + i
            for key in loss_dict.names:
                writer.add_scalar(logging_mode + '_' + key, loss_dict.get(key).val,
                                  step)
        
    if args.tensorboard:
        
        if not model.training:
            for dices in columns:
                writer.add_scalar(logging_mode + '_' + dices + '_avg',
                              dice_dataframe.mean()[dices], epoch)

        # Add average value to the tensorboard
        avg_dict = loss_dict.return_all_avg()
        for key in ['MSE_loss', 'Regu_Loss', 'Deformed_mask_Loss']:
            if key in avg_dict:
                writer.add_scalar(logging_mode + '_' + key + '_avg', 
                                  avg_dict[key], epoch)
       
        if epoch % args.image_tensorboard_frequency == 0:
            # Add images to tensorboard
            n = moving.shape[0] 
            for batch in range(n):
                mse_loss = mse_loss if args.plot_loss else None
                deformed_moving_mask = deformed_moving_mask if args.plot_mask else None
                
                fig_registration = ImageTensorboard.plot_registration_results(gt_sample, 
                                                                              pred_sample,
                                                                              batch, 
                                                                              args, 
                                                                              moving_patient, 
                                                                              reference_patient, 
                                                                              mse_loss,
                                                                              deformed_moving_mask)

                writer.add_figure(logging_mode + str(batch) + '_Regis', 
                                  fig_registration, epoch)

    return loss_dict.get('Loss').avg


if __name__ == '__main__':

    debug = False

    if debug:
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        call(["nvcc", "--version"])
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call(["nvidia-smi", "--format=csv",
              "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', torch.cuda.current_device())

        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
        

    
    parser = parse_args()
    args = parser.parse_args()
    main(args)
