# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:39:16 2020

@author: T_ESTIENNE
"""
import torch
from collections import OrderedDict

repo = 'abdominal_registration'
main_path = './' + repo + '/'

networks_path = main_path
load_model_path = main_path + '/save/models/'


# %% Creation of model

def create_model(args, kwargs):
    '''
        Dynamic creation of a network
    '''
    model_type = args.arch

    package_name = model_type
    package = __import__(repo + '.' + package_name)

    network_package = getattr(package, model_type)

    model = getattr(network_package, model_type)(**kwargs)

    print('Create {} model'.format(model_type))
    return model

# %%

def handleDataParallel(checkpoint):
    '''
        If the original data is save with DataParallel 
        we need to create a new OrderedDict that does not contains 
        'module'
    '''
    first_key = checkpoint.keys().__iter__().__next__()
    # Check if the module was saved with DataParallel
    if first_key.startswith('module'):
        print('handleDataParallel')
        new_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            name = key[7:]  # remove 'module'
            new_checkpoint[name] = value
        return new_checkpoint
    else:
        return checkpoint


def load_model(args, kwargs):
    '''
        This function load a pretrained model
    '''
    file = args.model_abspath
    name = args.model_abspath.split('/')[-1]

    # Model
    model = create_model(args, kwargs)

    print("=> loading model '{}'".format(name))
    checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
        best_loss = checkpoint['val_loss']
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    state_dict = handleDataParallel(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    if 'epoch' in checkpoint.keys():
        print("=> loaded model '{}' (epoch {} / val_loss {}"
            .format(name, epoch, best_loss))

    return model, name


def load_pretrained_model(args, kwargs):
    '''
        This function load a pretrained model
    '''
    file = args.model_abspath
    name = args.model_abspath.split('/')[-1]

    print_ = True
    # print_ = False

    # Model
    model = create_model(args, kwargs)

    print("=> loading model '{}'".format(name))
    checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    epoch = checkpoint['epoch']
    best_loss = checkpoint['val_loss']
    pretrained_dict = handleDataParallel(checkpoint['state_dict'])
    
    model_dict = model.state_dict()

    if print_:
        print()
        print('Model')
        for param_tensor in model_dict:
            print(param_tensor, "\t", model_dict[param_tensor].size())

        print()
        print(model)

        print('Pretrained dict')
        for param_tensor in pretrained_dict:
            print(param_tensor, "\t", pretrained_dict[param_tensor].size())
        print()


    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}

    if print_:
        print()
        print('Intersection dict')
        for param_tensor in pretrained_dict:
            print(param_tensor, "\t", pretrained_dict[param_tensor].size())

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("=> loaded model '{}' (epoch {} / val_loss {}"
          .format(name, epoch, best_loss))

    return model, name
