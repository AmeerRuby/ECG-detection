# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:26:29 2019

@author: Arcy
"""

#
import torch
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

#
def build_dataloader(X, y, resamp=True, batch_size=128):
    
    print("Start building dataloader.")
    df = pd.DataFrame(data=np.hstack((X, y)), 
                      columns=[i for i in range(X.shape[1])] + ['Class'])
    
    y = df.Class
    X = df.drop('Class', axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15, random_state=77)
    
    if resamp == True:
        X_train, y_train = resample_data(X_train, y_train)
        #X_val, y_val = resample_data(X_val, y_val)
    
    X_train = X_train.values
    X_val = X_val.values
    y_train = y_train.values.astype('int')
    y_val = y_val.values.astype('int')
    
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)
    
    train = torch.utils.data.TensorDataset(X_train, y_train)
    val = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader

#
def resample_data(data, label):
    
    X = pd.concat([data, label], axis=1)
    afib = X[X.Class == 0]
    gsvt = X[X.Class == 1]
    sb = X[X.Class == 2]
    sr = X[X.Class == 3]

    # Downsample GSVT, SB, and SR to match the number of samples in AFIB
    gsvt_new = resample(gsvt, replace=False, n_samples=len(afib), random_state=77)
    sb_new = resample(sb, replace=False, n_samples=len(afib), random_state=77)
    sr_new = resample(sr, replace=False, n_samples=len(afib), random_state=77)

    # Concatenate the downsampled data with the original AFIB class
    df_n = pd.concat([afib, gsvt_new, sb_new, sr_new])
    y_ = df_n.Class
    X_ = df_n.drop('Class', axis=1)

    return X_, y_