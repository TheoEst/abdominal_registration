# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:43:25 2020

@author: T_ESTIENNE
"""
import torch
import numpy as np
import os
import math

repo = 'abdominal_registration/'
main_path = './' + repo


def to_var(args, x):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if args.cuda:
        x = x.cuda(args.gpu)
    elif args.data_parallel:
        x = x.cuda()
    return torch.autograd.Variable(x)


def to_numpy(args, x):
    if not (isinstance(x, np.ndarray) or x is None):
        if args.cuda:
            x = x.cpu()
        x = x.detach().numpy()

    return x


def save_checkpoint(args, state, is_best):
    '''
        Save the current model. 
        If the model is the best model since beginning of the training
        it will be copy
    '''
    save_path = args.model_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    epoch = state['epoch']
    
    if args.save and epoch % args.save_frequency == 0:    
        val_loss = state['val_loss']
        filename = save_path + '/' + \
            'model.{:02d}--{:.3f}.pth.tar'.format(epoch, val_loss)
        torch.save(state, filename)
    
    if is_best:
        filename = save_path + '/model_best.pth.tar'
        torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not math.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def return_string(self):
        return '{loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=self)


class MultiAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, names=[]):
        self.average_dict = {}
        self.names = names
        for name in self.names:
            self.average_dict[name] = AverageMeter()

    def get(self, name):
        return self.average_dict[name]

    def update(self, name, val, n=1):
        if name not in self.names:
            self.names.append(name)
            self.average_dict[name] = AverageMeter()
        self.get(name).update(val, n)

    def return_string(self):
        string = ''
        for name in self.names:
            string += (str(name) + ' ' +
                       self.get(name).return_string() + '\t')
        return string

    def update_Logger(self, Logger, epoch):
        for name in self.names:
            Logger.log_value(name, self.get(name).avg, epoch)

        return Logger
    
    def return_all_avg(self):
        return {name : self.average_dict[name].avg for name in self.names}


def print_summary(epoch, i, nb_batch, loss_dict, logging, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    if isinstance(loss_dict, MultiAverageMeter):
        string += loss_dict.return_string()
    else:
        for loss_name, loss in loss_dict.items():
            string += (loss_name + ' {:.4f} \t').format(loss)

    summary += string

    logging.info(summary)
