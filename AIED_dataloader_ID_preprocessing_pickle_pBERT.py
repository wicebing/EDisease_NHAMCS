import os
import time
import unicodedata
import random
import string
import re
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertTokenizer
import torch
import AIED_dataloader_v21 as AIED_dataloader
import AIED_bert_v21 as AIED_bert
import AIED_utils
from AIED_utils import save_checkpoint,load_checkpoint,count_parameters
from ranger import Ranger
from AIED_utils import draw_tsne_v2 as draw_tsne

import pickle

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test'

batch_size = 32
device = 'cuda'
parallel = False

checkpoint_file = './checkpoint_emb'

def debug_writer(debug,tp=0):
    import csv
    if tp == 0:
        filename = 'debug.txt'
    elif tp ==1 :
        filename = 'largeLossDebug.txt'
    elif tp ==2 :
        filename = 'debugnanloss.txt'
    pdebug = pd.DataFrame(debug)
    pdebug.to_csv(filename)
    print('show debug',tp)

def train_AIemb(DS_model,
                dim_model,
                baseBERT,
                dloader,
                normalization,
                dataset,
                sl=0,
                noise_scale=0.002,
                mask_ratio=0.33,
                lr=1e-5,
                epoch=100,
                log_interval=10,
                parallel=parallel):
    global checkpoint_file
    DS_model.to(device)
    dim_model.to(device)
    baseBERT.to(device)
    baseBERT.eval()
    DS_model.eval()
    dim_model.eval()
    
    #model_optimizer = Ranger(DS_model.parameters(),lr=lr)
    #model_optimizer_dim = Ranger(dim_model.parameters(),lr=lr) 
#     model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
#     model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if device == 'cuda':
            torch.cuda.set_device(0)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    
    for ep in range(1):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,normalization,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=True)

            sidx = sample['idx']

            for j, idx in enumerate(sidx):
                dsidx= dataset[dataset['idx']==idx.item()].index[0]
                dataset.at[dsidx,'ccemb'] = c_emb[j].cpu()
                dataset.at[dsidx,'hxemb'] = h_emb_mean[j].cpu()
                dataset.at[dsidx,'piemb'] = p_emb[j].cpu()

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        0))  

    pklfile = 'trainset_pickle_pi_evm_pBERT_3.pickle'

    try:
        with open(pklfile,'wb') as f:
            pickle.dump(dataset,f)
            print('** complete create pickle **')
    except:
        pass 
    return dataset    
    
config = {'hidden_size': 96,
          'bert_hidden_size': 768,
          'max_position_embeddings':512,
          'eps': 1e-12,
          'input_size': 64,
          'vocab_size':64,
          'type_vocab_size':4,
          'hidden_dropout_prob': 0.1,
          'num_attention_heads': 4, 
          'attention_probs_dropout_prob': 0.2,
          'intermediate_size': 64,
          'num_hidden_layers': 3,
          'structure_size':12,
          'order_size':256
         }

pretrained_weights="bert-base-multilingual-cased"
BERT_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
emb_model = AIED_bert.ewed_Model(config=config,
                                 tokanizer=BERT_tokenizer,
                                 device=device)
dim_model = AIED_bert.DIM(config=config,
                          device=device,
                          alpha=1,
                          beta=1,
                          gamma=1e-6)

expand_model = AIED_bert.ewed_expand_Model(config=config)

cls_model = AIED_bert.ewed_CLS_Model(config=config,device=device)

print('emb_model PARAMETERS: ' ,AIED_bert.count_parameters(emb_model))
print('dim_model PARAMETERS: ' ,AIED_bert.count_parameters(dim_model))
print('cls_model PARAMETERS: ' ,AIED_bert.count_parameters(cls_model))

baseBERT = AIED_bert.bert_baseModel()
print(' ** confirm pretrained BERT from_pretrained BERT ** ')

try: 
    baseBERT = load_checkpoint(checkpoint_file,'BERT_ml_pretrain2.pth',baseBERT)
    print(' ** Complete Load baseBERT Model ** ')
except:
    print('*** No Pretrain_baseBERT_Model ***')
for param in baseBERT.parameters():
    param.requires_grad = False   
print(' ** pretrained BERT WEIGHT ** ')

print('baseBERT PARAMETERS: ' ,AIED_bert.count_parameters(baseBERT))

success_load = 0

try: 
    emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
    print(' ** Complete Load EDisease Model ** ')
except:
    print('*** No Pretrain_EDisease_Model ***')
    success_load = 1

try:     
    dim_model = load_checkpoint(checkpoint_file,'DIM.pth',dim_model)
except:
    print('*** No Pretrain_DIM_Model ***')
    pass

try:     
    cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
except:
    print('*** No Pretrain_cls_Model ***')
    pass

all_datas = AIED_dataloader.load_datas()

data15_triage_train = all_datas['data15_triage_train']
data01_person = all_datas['data01_person']
data02_wh = all_datas['data02_wh']
data25_diagnoses = all_datas['data25_diagnoses']
dm_normalization_np = all_datas['dm_normalization_np']
data07_death = all_datas['data07_death']
data25_icu = all_datas['data25_icu']

EDEW_DS = AIED_dataloader.EDEW_Dataset(ds=data15_triage_train,
                                       tokanizer = BERT_tokenizer,
                                       data01_person = data01_person,
                                       data02_wh = data02_wh,
                                       data25_diagnoses= data25_diagnoses,
                                       data25_icu=data25_icu,
                                       data07_death = data07_death,
                                       normalization = dm_normalization_np,
                                       supervise=False,
                                       hxfile=False
                                      )

EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=32,
                         batch_size=batch_size,
                         collate_fn=AIED_dataloader.collate_fn)

structure_mean = dm_normalization_np[0]
structure_std  = dm_normalization_np[1]

print('dm_mean', structure_mean)
print('dm_std', structure_std)    

print('batch_size = ', batch_size)

data15_triage_train['ccemb']=None
data15_triage_train['hxemb']=None
data15_triage_train['piemb']=None

dataset = train_AIemb(DS_model=emb_model,
            dim_model=dim_model,
            baseBERT=baseBERT,
            dloader=EDEW_DL,
            normalization=structure_std,                
            noise_scale=0.001,
            mask_ratio=0.33,
            lr=4e-5,
            epoch=100,
            log_interval=15,
            parallel=parallel,
            sl=success_load,
            dataset=data15_triage_train)
    

