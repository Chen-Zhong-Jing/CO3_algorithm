# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:24:06 2022

@author: USER
"""
import numpy as np
from scipy.integrate import quad
from scipy import stats

def pdf_doubleweibull(x, a, m, scale=1):
  return stats.dweibull.pdf(x,a,m,scale)

def dweibullquantizer(data, R, iterations_kmeans, QUANTIZATION_M, cache_table):
    M = QUANTIZATION_M
    #M = 0
    mu = np.mean(data)
    std_threshold = 0.1 ** 6
    s = np.var(data)
    std = np.sqrt(s)
    if std < std_threshold:
        std = std_threshold
    
    data_normalized = np.nan_to_num(np.divide(np.subtract(data,mu),std))
    a, m, b = stats.dweibull.fit(data_normalized)
    cacheKey = (np.around(a,decimals = 2),np.around(b,decimals = 2))
    if cacheKey in cache_table:
        return cache_table[cacheKey]["threshold"], cache_table[cacheKey]["quantization_centers"]
    #b = 1
    #print(cacheKey)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    cache_table[cacheKey] = {"threshold": np.sort(np.add(np.multiply(thresholds_update,std),mu)),"quantization_centers": np.add(np.multiply(centers_update,np.sqrt(s)),mu)}
    return  cache_table[cacheKey]["threshold"], cache_table[cacheKey]["quantization_centers"]
