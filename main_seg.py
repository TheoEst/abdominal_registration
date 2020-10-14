# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:04:43 2020

@author: t_estienne
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

repo = 'abdominal_registration/'
main_path = './' + repo
model_names = ['SegNet']

cohorts = ['liver', 'pancreas', 'spleen', 'colon', 
           'learn2reg_task3', 'tcia_pancreas', 'kits', 'hepatic']

def parse_args(add_help=True):

    parser = argparse.ArgumentParser(
        description='Pytorch automatic registration',
        add_help=add_help)

    parser.add_argument('--arch', '-a', metavar='ARCH', default='SegNet',
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
    parser.add_argument('--save-frequency', type=int, default=5,
                        help='Save the model every X epoch')
    parser.add_argument('--image-tensorboard-frequency', type=int, default=3,
                        help='Plot the model in tensorboard every X epoch')
    parser.add_argument('--channel-multiplication', type=float, default=4,
                        help='Divide the number of channels of each convolution')
    parser.add_argument('--pool-blocks', type=int, default=4,
                        help='Number of pooling block (Minimum 2)')
    parser.add_argument('--channels', type=int, default=None, nargs='+',
                        help='List of the channels')
    parser.add_argument('--random-crop', action='store_true',
                        help='Do random crop')
    parser.add_argument('--data-augmentation', action='store_true',
                        help='Add axial flip and data augmentation')
    parser.add_argument('--crop-size', type=int, nargs='+', default=64,
                        help='Crop size')
    parser.add_argument('--val-crop-size', type=int, nargs='+', default=None,
                        help='Crop size')
    parser.add_argument('--instance-norm', action='store_true',
                        help='Use instance norm during training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='Use batch norm during training')
    parser.add_argument('--nb-Convs', type=int, default=None, nargs='+',
                        help='List of the channels')
    parser.add_argument('--classic-vnet', action='store_true', default=None,
                        help='Use classic')
    parser.add_argument('--model-abspath', type=str, default=None,
                        help='Absolute path of model to load')
    parser.add_argument('--cohorts', type=str, nargs='+', choices=cohorts,
                        default=['learn2reg_task3'], help='Cohort to use')
    parser.add_argument('--val-cohorts', type=str, nargs='+', choices=cohorts,
                        default=[], help='Cohort to use')

    return parser


def main(args):
    
    args.main_path = main_path
    args.val_cohorts = args.cohorts if len(args.val_cohorts) == 0 else args.val_cohorts
    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    args.merge_train_val, args.test = False, False
    args.inference = False

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

    if args.arch in ['SegNet']:
        params = ['channel_multiplication', 'pool_blocks', 'channels',
                  'instance_norm', 'batch_norm', 'nb_Convs']

        for param in params:
            model_kwargs[param] = getattr(args, param)
    
    if args.model_abspath is not None:
        
        (model, 
         model_epoch) = model_loader.load_model(args, model_kwargs)
    else:
        model = model_loader.create_model(args, model_kwargs)
        
    logging.info('=> Model ready')
    logging.info(model)

    # Loss
    criterion = {'seg': losses.masked_mean_dice_loss
                 }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data        
    if args.random_crop:
        crop = transformations.RandomCrop(args.crop_size)
    else:
        crop = transformations.CenterCrop(args.crop_size)
            
    val_crop = transformations.CenterCrop(val_crop_size)

    transforms_list = [transformations.Normalize(),
                       crop]
    
    if args.data_augmentation:
        transforms_list.append(transformations.AxialFlip())
        transforms_list.append(transformations.RandomRotation90())
        
        
    val_transforms_list = [transformations.Normalize(),
                           val_crop]
    

    transformation = torchvision.transforms.Compose(transforms_list)
    val_transformation = torchvision.transforms.Compose(val_transforms_list)

    (train_Dataset, val_Dataset) = Dataset.init_datasets(transformation,
                                                         val_transformation,
                                                         args,
                                                         segmentation=True)


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
    
    
    if args.data_parallel:
        model = nn.DataParallel(model).cuda(args.gpu)

    elif args.cuda:
        model = model.cuda(args.gpu)

    # summary(model, input_size=(4, *args.crop_size))
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


def train(loader, model, criterion, optimizer, writer,
          logging, epoch, args):

    end = time.time()
    loss_dict = utils.MultiAverageMeter([])

    logging_mode = 'Train' if model.training else 'Val'
    Dice = criterion['seg']
    
    columns = ['Spleen', 'Right_Kidney', 'Left_Kidney', 'Gall_Bladder',
               'Esophagus', 'Liver', 'Stomach', 'Aorta', 
               'Inferior_Vena_Cava', 'Portal&Splenic_Vein', 
               'Pancreas']

    dice_dataframe = pd.DataFrame(columns=columns)
        
    for i, gt_sample in enumerate(loader, 1):

        # measure data loading time
        data_time = time.time() - end
        patients = gt_sample['patient']
        
        (ct, mask_gt, label) = (gt_sample['ct'], gt_sample['mask'], 
                                gt_sample['label'])
        
        ct = utils.to_var(args, ct.float())
        mask_gt = utils.to_var(args, mask_gt.float())
        
        # compute output
        mask_pred = model(ct)

        # compute loss
        seg_loss = Dice(mask_pred, mask_gt, label)
        loss_dict.update('Seg_loss', seg_loss.item())
        loss = seg_loss
        loss_dict.update('Loss', loss.item())
        
        if model.training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Metrics        
        if not model.training:
            patients = gt_sample['patient']
            
            dataframe = losses.evalAllSample(mask_pred, mask_gt, 
                                             patients, args)
            dice_dataframe = pd.concat([dice_dataframe, dataframe])
            
            for dices in columns:
                if dices in dataframe.mean().index:
                    loss_dict.update(dices, dataframe.mean()[dices])
                
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
        for key in ['Seg_loss']:
            if key in avg_dict:
                writer.add_scalar(logging_mode + '_' + key + '_avg', 
                                  avg_dict[key], epoch)
       
        if epoch % args.image_tensorboard_frequency == 0:
            # Add images to tensorboard
            n = ct.shape[0]
            for batch in range(n):
                
                fig_segmentation = ImageTensorboard.plot_segmentation_ct(ct, mask_gt, 
                                                                        mask_pred,
                                                                        batch, 
                                                                        patients,
                                                                        args)
                
                writer.add_figure(logging_mode + str(batch) + '_Seg', 
                                  fig_segmentation, epoch)

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
