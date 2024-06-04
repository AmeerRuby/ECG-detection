# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:22:01 2019
@author: Arcy
"""
#%%
import torch
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from utils_test import get_ecg, qrs_detection, get_segments, plot, plot_confusion_matrix
from data import build_dataloader
from train_test import learn, cnn_feed_lstm

PATH = "/Users/nguyenvietthai/Downloads/Datamoi/mat_files"
BATCH_SIZE = 2048
EPOCH = 50
FS = 300
LENGTH = 9000
LR = 1e-3
RESAMP = False

#%%
try:
    segments = np.load('/Users/nguyenvietthai/Downloads/Datamoi/mat_files/segment.npy')
except FileNotFoundError:
    print("File not found. Generating segments...")
    PATH = '/Users/nguyenvietthai/Downloads/Datamoi/mat_files' 
    LENGTH = 9000
    signals, labels = get_ecg(PATH, segment_length=LENGTH)
    
    if len(signals) == 0:
        raise ValueError("No signals found. Please check the PATH and LENGTH parameters.")
    
    segments = np.zeros((245990, 1001))
    k = 0
    
    for i, record in enumerate(signals):
        print(f"Running record {i + 1} of {len(signals)}")
        rp = qrs_detection(record, sample_rate=FS)
        seg = get_segments(record, rp, labels[i])
        
        if seg is not None:
            segments[k:k+seg.shape[0], :] = seg
            k += seg.shape[0]
    
    del signals, labels
    
    np.save('/Users/nguyenvietthai/Downloads/Datamoi/mat_files/segment.npy', segments)
#%%
X, y = segments[:, :-1], segments[:, -1][:, np.newaxis]
del segments

# Split data into train, validation, and test sets
train, test = build_dataloader(X, y, resamp=RESAMP, batch_size=BATCH_SIZE)
del X, y

net = cnn_feed_lstm()
try:
    params = torch.load("../params/net_0.81.pkl")
    net.load_state_dict(params["model_state_dict"])
except:
    pass
#%%
loss, val_score, test_score, test_predictions, test_true_labels = learn(net, train, test, lr=LR, epoch=EPOCH)
plot(loss, val_score, test_score)

#%%
cm = confusion_matrix(test_true_labels, test_predictions)
classes = ['AFIB', 'GSVT', 'SB', 'SR']

# Plot the confusion matrix
plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')




# %%
