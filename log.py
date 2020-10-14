# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:49:42 2020

@author: T_ESTIENNE
"""
import logging

def set_logger(log_path):

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s :: %(levelname)s ::  %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def clear_logger(log):

    for handler in log.handlers:
        log.removeHandler(handler)

    return

