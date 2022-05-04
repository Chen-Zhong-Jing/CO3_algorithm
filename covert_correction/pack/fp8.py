# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:20:48 2022

@author: Zhong_Jing
"""
import numpy as np

def fp8_143_bin_edges(exponent_bias=10):
    bin_centers = np.zeros(239,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(239, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2, 2 ** -3],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(15):
            for f in range(8):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(4), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f, 'b').zfill(3)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f,'b').zfill(3)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict

def fp8_152_bin_edges(exponent_bias=15):
    bin_centers = np.zeros(247,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(247, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(31):
            for f in range(4):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(5), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(5) + format(f, 'b').zfill(2)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(5) + format(f,'b').zfill(2)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict
