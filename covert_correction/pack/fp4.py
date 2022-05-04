# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:56:26 2022

@author: Zhong_Jing
"""
import numpy as np
def fp4_121_bin_edges(exponent_bias=1):
    bin_centers = np.zeros(11,dtype=np.float32)
    binary_dict = {}
    binary_fraction = np.array([2 ** -1],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(3):
            for f in range(2):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(2), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(1)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(2) + format(f, 'b').zfill(1)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(1)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(2) + format(f,'b').zfill(1)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    #bin_edges = np.hstack((-np.inf,bin_edges))
    #bin_edges = np.append(bin_edges,np.inf)
    return bin_centers, bin_edges, binary_dict

def fp4_best_scalar(params):
    x = params[0]
    scalar = (-2.85 * x) + (5.37 * x**2) + (-2.85 * x**3) + (0.52 * x**4) + 0.46
    if scalar < 0.0009:
        scalar = 0.0009/params[2]
    else:
        scalar /= params[2]
        
    return scalar