import os
import time
import unicodedata
import random
import string
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle

import matplotlib.pyplot as plt


pklfile = 'ROC_results.pickle'

with open(pklfile,'rb') as f:
    results = pickle.load(f)  
print('** complete load pickle **')

    # result ={'target':f_target.numpy(),
    #          'idx':idx.cpu().numpy(),
    #          'label':labels,
    #          'pred':pred}

def convert_keys(k):
    if k == 'ub':
        return 'Ours'
    elif k == 'dc':
        return 'Ours, small'
    elif k == 'dim':
        return 'Only DIM, small'
    elif k == 'clr':
        return 'Only SimCLR, small'
    elif k == 'none':
        return 'Without pretrained, small'


def plot_roc_nhamcs2(results,ep=0):
    from sklearn.metrics import roc_curve, auc
    ROC_threshold = torch.linspace(0,1,100).numpy()

    fig = plt.figure(figsize=(6,6),dpi=200)
    ax = fig.add_subplot(111)
    
    for k,v in results.items():
        labels = v['label']
        pred = v['pred']
        
        all_label = labels['all_label']
        
        icu_tlabel_ = all_label[:,3]
        die_tlabel_ = all_label[:,4]
        
        age = all_label[:,5]
        
        adult = age >= 18
        child = age <= 18
        sep = [adult,child]
     
        icu_tlabel = []
        for a in icu_tlabel_:
            icu_tlabel.append(int(a<8))
        icu_tlabel = np.array(icu_tlabel)
        
        die_tlabel = []
        for a in die_tlabel_:
            die_tlabel.append(int(a<8))
        die_tlabel = np.array(die_tlabel)
        
        poor_tlabel = []
        for i,a in enumerate(icu_tlabel_):
            if icu_tlabel_[i]<8 or die_tlabel_[i]<8:
                poor_tlabel.append(1)
            else:
                poor_tlabel.append(0)
        poor_tlabel = np.array(poor_tlabel)    
        
        pred_poor = pred['poor']
        
        fpr, tpr, _ = roc_curve(poor_tlabel, pred_poor.numpy())
        roc_auc2 = auc(fpr,tpr)
        
        s = convert_keys(k)
        label_auc2 = str(s)+', AUROC='+str(round(roc_auc2,3))        
        ax.plot(fpr,tpr,label=label_auc2)
        
    ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')

    ax.set_xlabel('1-specificity')
    ax.set_ylabel('sensitivity')
    ax.set_title('ROC curves')
    plt.legend()
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    plt.legend()
    plt.savefig('./pic_ROC/AUCs_nhamcs.png')
