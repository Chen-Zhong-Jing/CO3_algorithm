# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 19:06:11 2022

@author: Zhong_Jing
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
from scipy import stats
from scipy.optimize import minimize_scalar

def W(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u)[1:u.size - 1] - np.sort(v)[1:v.size - 1]) ** p) ** (1 / p)

#Layerlist = [4, 36, 140]
#Layerlist = [95, 326, 695]
Layerlist = [5, 44, 266]
nEpoch = 150
nSim = 2
nSample = 3000

WassDist = np.zeros((len(Layerlist),nEpoch,3))
ParamGenArr = np.zeros((len(Layerlist),nEpoch,3))
for e in range(nEpoch):
    for s in range(nSim):
        FilePath = 'G:\\.shortcut-targets-by-id\\1-COUiwl8Mq0au-4Eau2TBM0shrbV8_2s\\Simulations\\CIFAR10\\DenseNet121\\SGD\\' + str(s) + '\\'
        data = pickle.load(open(FilePath + 'Gradient_epoch' + str(e) + '.p', "rb"))
        for l in range(len(Layerlist)):
            Grad = data['avr_grad'][Layerlist[l]].flatten()
            GradStd = np.std(Grad)
            GradMean =  np.mean(Grad)
            Grad = (Grad - GradMean) / GradStd

            GradSample = np.random.choice(Grad, nSample)

            paramGau = stats.norm.fit(Grad)
            WassDist[l, e, 0] += W(2, stats.norm.ppf(np.linspace(0, 1, nSample), loc=paramGau[0], scale=paramGau[1]),np.sort(GradSample))
            
            paramLap = stats.laplace.fit(Grad)
            WassDist[l, e, 1] += W(2, stats.laplace.ppf(np.linspace(0, 1, nSample), loc=paramLap[0], scale=paramLap[1]),np.sort(GradSample))
            
            ParamGenArr[l,e] = stats.gennorm.fit(Grad)
            WassDist[l, e, 2] += W(2, stats.gennorm.ppf(np.linspace(0, 1, nSample), ParamGenArr[l,e,0], loc=ParamGenArr[l,e,1],scale=ParamGenArr[l,e,2]), np.sort(GradSample))



WassDist /= nSim
ParamGenArr /= nSim
info = {'ParamGenNorm': ParamGenArr, 'WassersteinDistance': WassDist}
filepath = 'D:\\JSAC\\'
strModel = "DenseNet121"
pickle.dump(info, open(filepath + strModel + '.p', "wb"))

plt.figure()
plt.plot(WassDist[0,:,0], '-*')
plt.plot(WassDist[0,:,1], '-*')
plt.plot(WassDist[0,:,2], '-*')
plt.xlabel('epoch')
plt.ylabel('W2 distance')
plt.legend(['Normal','Laplace', 'GenNorm'])
plt.title(strModel + '-Upper')

plt.figure()
plt.plot(WassDist[1,:,0], '-*')
plt.plot(WassDist[1,:,1], '-*')
plt.plot(WassDist[1,:,2], '-*')
plt.xlabel('epoch')
plt.ylabel('w2 distance')
plt.legend(['normal','Laplace', 'GenNorm'])
plt.title(strModel +'-Middle')

plt.figure()
plt.plot(WassDist[2,:,0], '-*')
plt.plot(WassDist[2,:,1], '-*')
plt.plot(WassDist[2,:,2], '-*')
plt.xlabel('epoch')
plt.ylabel('w2 distance')
plt.legend(['normal', 'Laplace', 'GenNorm'])
plt.title(strModel +'-Lower')
plt.show()

