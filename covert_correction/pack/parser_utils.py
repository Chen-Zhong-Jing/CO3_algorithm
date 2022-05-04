# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:13:23 2021
@author: Eduin Hernandez
"""
import argparse

def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif string.lower() in ('none'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')        