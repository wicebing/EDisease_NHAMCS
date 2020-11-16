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
import AIED_dataloader_nhamcs as AIED_dataloader
import AIED_bert_v31 as AIED_bert
import AIED_utils
from AIED_utils import save_checkpoint,load_checkpoint,count_parameters
# from ranger import Ranger
from AIED_utils import draw_tsne_v2 as draw_tsne
from AIED_utils import draw_tsne_cls, count_tsne_cls
import pickle

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test_nhamcs_cls'

try:
    pi = sys.argv[2]
    print('*****pi= ',pi)
except:
    pi = '0'    
    
batch_size = 1024
device = 'cuda'
parallel = False

checkpoint_file = './checkpoint_emb/aRevision_dc'
alpha=1
beta=1
gamma=0.1

if pi=='1':
    use_pi =True
    nw = 3
else:
    use_pi =False
    nw = 5
    
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
    #baseBERT.eval()
    
    #model_optimizer = Ranger(DS_model.parameters(),lr=lr)
    #model_optimizer_dim = Ranger(dim_model.parameters(),lr=lr) 
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        dim_model = torch.nn.DataParallel(dim_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
            
            s_np = sample['structure'].numpy()
            c_np = sample['cc'].numpy()
            h_np = sample['ehx'].numpy()
            nans = np.all(np.isnan(s_np))
            nanc = np.all(np.isnan(c_np))
            nanh = np.all(np.isnan(h_np))
            
            ptloss = True if batch_idx%699==3 else False
            
            if ~nans and ~nanc and ~nanh:
                #try:
                output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean), nohx, expand_data = DS_model(baseBERT,sample,normalization,noise_scale=noise_scale,mask_ratio=mask_ratio)
                
                bs = len(s)
                if torch.all(torch.isnan(input_emb_org)):
                    # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                    debug_writer(debugnan,tp=2) 
                    continue                 
                loss = dim_model(output[:,:1],
                                 input_emb_org,
                                 SEP_emb_emb,
                                 nohx,
                                 mask_ratio=mask_ratio,
                                 mode=mode,
                                 ptloss=ptloss
                                )

                if str(loss.item())=='nan':
                    # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                    debug_writer(debugnan,tp=2)                       
                else:
                    loss.backward()
                    model_optimizer.step()
                    model_optimizer_dim.step()

                    with torch.no_grad():
                        epoch_loss += loss.item()*bs
                        epoch_cases += bs
                #print(11111111111)
                if iteration % log_interval == 0:
                    print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                            ep, batch_idx * batch_size,
                            100. * batch_idx / len(dloader),
                            (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                            loss.item()))

                #except:
                #    debug.append([sample['structure'],sample['cc'],sample['ehx']])
                #    debug_writer(debug)
                #    print('pass:', batch_idx)
            else:
                print('nans: ', nans)
                print('nanc: ', nanc)
                print('nanh: ', nanh)
                
            if iteration % 500 == 0:
                try:  
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            if iteration % 50000 == 5001:
                checkpoint_pathName = '{:.0f}_{:.0f}'.format(time.time(),sl)
                try:  
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_'+checkpoint_pathName+'.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM_'+checkpoint_pathName+'.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
                 
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune.csv', sep = ',')
    print(total_loss)

def train_pickle(DS_model,
                 dim_model,
                 baseBERT,
                 dloader,
                 sl=0,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 lr=1e-4,
                 epoch=100,
                 log_interval=10,
                 parallel=parallel,
                 mix_ratio=1,
                 use_pi=False):
    global checkpoint_file
    DS_model.to(device)
    dim_model.to(device)
    baseBERT.to(device)
    
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        dim_model = torch.nn.DataParallel(dim_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=use_pi)
            
            #augmentation 2 for simCLR
            aug2 = 2*random.random()
            output2,EDisease2, (s,input_emb2,input_emb_org2), _,_, _, _ = DS_model(baseBERT,sample,noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=use_pi)
                       
            bs = len(s)
             
            loss = dim_model(output[:,:1],
                             input_emb_org,
                             CLS_emb_emb,
                             nohx,
                             mask_ratio=mask_ratio,
                             mode=mode,
                             ptloss=ptloss,
                             DS_model=DS_model,
                             mix_ratio=mix_ratio,
                             EDisease2=output2[:,:1],
                             shuffle=True,
                             use_pi=use_pi,
                             yespi=yespi,
                             ep=ep
                            )

            if str(loss.item())=='nan':
                debugnan.append(sample['idx'])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.backward()
                model_optimizer.step()
                model_optimizer_dim.step()

                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs
            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item()))

                
            if iteration % 500 == 0:
                try:  
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            if iteration % 50000 == 5001:
                checkpoint_pathName = '{:.0f}_{:.0f}'.format(time.time(),sl)
                try:  
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_'+checkpoint_pathName+'.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM_'+checkpoint_pathName+'.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
                 
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune.csv', sep = ',')
    print(total_loss)    
    
    
def test_AIemb(DS_model,baseBERT,dloader,parallel=parallel,device=device,use_pi=False,tsne=True):
    f_edisease = draw_tsne(net_F=DS_model,baseBERT=baseBERT, target=dloader, device=device,use_pi=use_pi,tsne=tsne)
    return f_edisease

def test_AIcls(DS_model,baseBERT,cls_model,dloader,parallel=parallel,device=device,use_pi=False,tsne=True):
    
    f_edisease = draw_tsne_cls(net_F=DS_model,
                               baseBERT=baseBERT,
                               cls_model=cls_model,
                               target=dloader,
                               device=device,
                               use_pi=use_pi,
                               tsne=tsne)
    return f_edisease

def count_AIcls(DS_model,baseBERT,cls_model,dloader,ki=0, parallel=parallel,device=device,use_pi=False,draw=False):
    
    f_edisease, auc = count_tsne_cls(net_F=DS_model,
                               baseBERT=baseBERT,
                               cls_model=cls_model,
                               target=dloader,
                               device=device,
                               use_pi=use_pi,
                               ki=ki,draw=draw)
    return f_edisease, auc

def make_trg(sample, train_cls=True):
    s_ = sample['structure']
    trg = sample['trg']
    '''
        trg = e_patient[['COMPUTEREDTRIAGE',
                         'TRIAGE',
                         'Hospital',
                         'icu7',
                         'death7',
                         'Age',
                         'Sex',
                         'cva',
                         'trauma',
                         'query']]     '''
    trg = trg.long()
    trg_triage_ = trg[:,1]
    trg_hospital = trg[:,2]
    trg_icu7_ = trg[:,3]
    trg_die7_ = trg[:,4]
    trg_cva = trg[:,7]
    
    age = trg[:,5]
    sex = trg[:,6]
    
    trg_triage = (trg_triage_<3).long()

    
    temp_icu = trg_icu7_<8
    temp_die = trg_die7_<8
    trg_icuANDdie7 = temp_icu | temp_die
    
    trg_icu7 = temp_icu.long()
    trg_die7 = temp_die.long()    
    trg_icuANDdie7 = trg_icuANDdie7.long()

#    age_ = (s_[:,0]*28.31)+43.71
#    sex_ = s_[:,1]
    
#    age = (age_/10).int()
#    sex = (sex_>0.5).int()
                
    trg_cls = {'trg_triage':trg_triage,
               'trg_icu7':trg_icu7,
               'trg_die7':trg_die7,
               'trg_icudeath7':trg_icuANDdie7,
               'cls_hospital':trg_hospital,
               'cls_cva':trg_cva,
               'cls_age':age,
               'cls_sex':sex
              }
        
    return trg_cls
    
def train_pickle_cls(DS_model,
                     cls_model,
                     baseBERT,
                     dloader,
                     checkpoint_file,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=1e-4,
                     epoch=100,
                     log_interval=10,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     use_pi=False):
    DS_model.to(device)
    cls_model.to(device)
    baseBERT.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(cls_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        cls_model = torch.nn.DataParallel(cls_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            if batch_idx > int(len(dloader)/5):
                break
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=use_pi)    

            bs = len(s)
            
            #using the hidden layer as EDisease, output for pretrain
            hidden = output
            cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])

            trg_cls=make_trg(sample, train_cls=True)                
            trg_triage=(trg_cls['trg_triage']).to(DS_model.device)
            trg_icu7=(trg_cls['trg_icu7']).to(DS_model.device)
            trg_die7=(trg_cls['trg_die7']).to(DS_model.device)
            trg_poor7=(trg_cls['trg_icudeath7']).to(DS_model.device)

            loss_icu = criterion(cls_icu,trg_icu7)
            loss_die = criterion(cls_die,trg_die7)
            loss_tri = criterion(cls_tri,trg_triage)
            loss_poor = criterion(cls_poor,trg_poor7)
            
            loss = loss_icu+loss_die+loss_tri+loss_poor

            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.backward()
                model_optimizer_cls.step()
                if trainED:
                    model_optimizer.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} Li:{:.4f} Ld:{:.4f} Lt:{:.4f} Lp:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item(), loss_icu.item(), loss_die.item(), loss_tri.item(),loss_poor.item()))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_CLS.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='CLS.pth',
                                    model=cls_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_CLS.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='CLS.pth',
                            model=cls_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        try:
            pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS_'+loss_pathName+'.csv', sep = ',')
        except:
            pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS.csv', sep = ',')
    print(total_loss)                

def train_NHAMCS_cls(DS_model,
                     cls_model,
                     baseBERT,
                     dloader,
                     checkpoint_file,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=1e-4,
                     epoch=100,
                     log_interval=10,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     use_pi=False):
    DS_model.to(device)
    cls_model.to(device)
    baseBERT.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(cls_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        cls_model = torch.nn.DataParallel(cls_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            # if batch_idx > int(len(dloader)/5):
            #     break
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            bs = len(s)
            
            #using the hidden layer as EDisease, output for pretrain
            hidden = output
            cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])

            trg_cls=make_trg(sample, train_cls=True)                
            trg_triage=(trg_cls['trg_triage']).to(DS_model.device)
            trg_icu7=(trg_cls['trg_icu7']).to(DS_model.device)
            trg_die7=(trg_cls['trg_die7']).to(DS_model.device)
            trg_poor7=(trg_cls['trg_icudeath7']).to(DS_model.device)
            
            cls_poor2 = torch.max(cls_icu,cls_die)

            loss_icu = criterion(cls_icu,trg_icu7)
            loss_die = criterion(cls_die,trg_die7)
            loss_tri = criterion(cls_tri,trg_triage)
            loss_poor = criterion(cls_poor,trg_poor7)
            
            loss = loss_poor + loss_icu + loss_die

            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.sum().backward()
                model_optimizer_cls.step()
                if trainED:
                    model_optimizer.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} Li:{:.4f} Ld:{:.4f} Lt:{:.4f} Lp:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item(), loss_icu.item(), loss_die.item(), loss_tri.item(),loss_poor.item()))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_CLS.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='CLS.pth',
                                    model=cls_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_CLS.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='CLS.pth',
                            model=cls_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)

        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path=f'EDisease_CLS_{ep}.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path=f'CLS_{ep}.pth',
                            model=cls_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS.csv', sep = ',')
    print(total_loss)    

def train_NHAMCS_cls_val(DS_model,
                     cls_model,
                     baseBERT,
                     dloader,
                     dloader_val,
                     checkpoint_file,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=1e-4,
                     epoch=100,
                     log_interval=10,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     use_pi=False):
    DS_model.to(device)
    cls_model.to(device)
    baseBERT.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(cls_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        cls_model = torch.nn.DataParallel(cls_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    aucs = []
    best_auc = 0
    
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            # if batch_idx > int(len(dloader)/5):
            #     break
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            bs = len(s)
            
            #using the hidden layer as EDisease, output for pretrain
            hidden = output
            cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])

            trg_cls=make_trg(sample, train_cls=True)                
            trg_triage=(trg_cls['trg_triage']).to(DS_model.device)
            trg_icu7=(trg_cls['trg_icu7']).to(DS_model.device)
            trg_die7=(trg_cls['trg_die7']).to(DS_model.device)
            trg_poor7=(trg_cls['trg_icudeath7']).to(DS_model.device)
            
            cls_poor2 = torch.max(cls_icu,cls_die)

            loss_icu = criterion(cls_icu,trg_icu7)
            loss_die = criterion(cls_die,trg_die7)
            loss_tri = criterion(cls_tri,trg_triage)
            loss_poor = criterion(cls_poor,trg_poor7)
            
            loss = loss_poor + loss_icu + loss_die

            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.sum().backward()
                model_optimizer_cls.step()
                if trainED:
                    model_optimizer.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} Li:{:.4f} Ld:{:.4f} Lt:{:.4f} Lp:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item(), loss_icu.item(), loss_die.item(), loss_tri.item(),loss_poor.item()))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_CLS.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='CLS.pth',
                                    model=cls_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1

        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_CLS.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='CLS.pth',
                            model=cls_model,
                            parallel=parallel)
            
            val_model = AIED_bert.ewed_Model(config=config,
                                 tokanizer=BERT_tokenizer,
                                 device=device)
            
            val_DS_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',val_model)
            
            _, auc = count_tsne_cls(net_F=val_DS_model,
                               baseBERT=baseBERT,
                               cls_model=cls_model,
                               target=dloader_val,
                               device=device,
                               use_pi=use_pi,
                               ki=ep,
                               draw=False)
            
            aucs.append([ep,auc])
            
            pd_aucs = pd.DataFrame(aucs)
            pd_aucs.to_csv('./loss_record/aucs.csv', sep = ',')
            
            cls_model.train()
            if auc > best_auc:
                # cls_model.train()
                save_checkpoint(checkpoint_file=checkpoint_file,
                                checkpoint_path='EDisease_CLS_BEST.pth',
                                model=DS_model,
                                parallel=parallel)
                save_checkpoint(checkpoint_file=checkpoint_file,
                                checkpoint_path='CLS_BEST.pth',
                                model=cls_model,
                                parallel=parallel)                
                
                best_auc = auc

            print(f'======= epoch:{ep} ======== ** best auc = {best_auc}')
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS.csv', sep = ',')
    print(total_loss)    

def train_NHAMCS(DS_model,
                 dim_model,
                 baseBERT,
                 dloader,
                 sl=0,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 lr=1e-4,
                 epoch=100,
                 log_interval=10,
                 parallel=parallel,
                 mix_ratio=1,
                 use_pi=False,                     
                 checkpoint_file=checkpoint_file,
                 ): 
    
    DS_model.to(device)
    dim_model.to(device)
    baseBERT.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        dim_model = torch.nn.DataParallel(dim_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            aug2 = 2*random.random()
            output2,EDisease2, (s,input_emb2,input_emb_org2), _,_, _, _ = DS_model(baseBERT,sample,noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=False)       
            
            bs = len(s)            

            loss = dim_model(output[:,:1],
                             input_emb_org,
                             CLS_emb_emb,
                             nohx,
                             mask_ratio=mask_ratio,
                             mode=mode,
                             ptloss=ptloss,
                             DS_model=DS_model,
                             mix_ratio=mix_ratio,
                             EDisease2=output2[:,:1],
                             shuffle=True,
                             use_pi=use_pi,
                             yespi=yespi,
                             ep=ep
                            )
            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.sum().backward()
                model_optimizer_cls.step()
                model_optimizer.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item()))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss.csv', sep = ',')
    print(total_loss) 

def train_NHAMCS_cls_dim(DS_model,
                     cls_model,
                     dim_model,
                     baseBERT,
                     dloader,
                     checkpoint_file,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     mix_ratio=1,
                     lr=1e-4,
                     epoch=100,
                     log_interval=10,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     use_pi=False):
    DS_model.to(device)
    cls_model.to(device)
    baseBERT.to(device)
    dim_model.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(cls_model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        cls_model = torch.nn.DataParallel(cls_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
        dim_model = torch.nn.DataParallel(dim_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            # if batch_idx > int(len(dloader)/5):
            #     break
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            model_optimizer_dim.zero_grad()
            
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            aug2 = 2*random.random()
            output2,EDisease2, (s,input_emb2,input_emb_org2), _,_, _, _ = DS_model(baseBERT,sample,noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=False)       
            
            bs = len(s)            

            loss_dim = dim_model(output[:,:1],
                             input_emb_org,
                             CLS_emb_emb,
                             nohx,
                             mask_ratio=mask_ratio,
                             mode=mode,
                             ptloss=ptloss,
                             DS_model=DS_model,
                             mix_ratio=mix_ratio,
                             EDisease2=output2[:,:1],
                             shuffle=True,
                             use_pi=use_pi,
                             yespi=yespi,
                             ep=ep
                            )

            hidden = output
            cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])

            trg_cls=make_trg(sample, train_cls=True)                
            trg_triage=(trg_cls['trg_triage']).to(DS_model.device)
            trg_icu7=(trg_cls['trg_icu7']).to(DS_model.device)
            trg_die7=(trg_cls['trg_die7']).to(DS_model.device)
            trg_poor7=(trg_cls['trg_icudeath7']).to(DS_model.device)
            
            cls_poor2 = torch.max(cls_icu,cls_die)

            loss_icu = criterion(cls_icu,trg_icu7)
            loss_die = criterion(cls_die,trg_die7)
            loss_tri = criterion(cls_tri,trg_triage)
            loss_poor = criterion(cls_poor,trg_poor7)
            
            loss = loss_poor + loss_icu + loss_die + loss_dim

            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.sum().backward()
                model_optimizer_cls.step()
                model_optimizer.step()
                model_optimizer_dim.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} Li:{:.4f} Ld:{:.4f} Lp:{:.4f} Ldim:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item(), loss_icu.item(), loss_die.item(), loss_poor.item(), loss_dim.item(),))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_CLS.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='CLS.pth',
                                    model=cls_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)                    
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_CLS.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='CLS.pth',
                            model=cls_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)

        if ep % 5 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path=f'EDisease_CLS_{ep}.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path=f'CLS_{ep}.pth',
                            model=cls_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS.csv', sep = ',')
    print(total_loss)  

def train_NHAMCS_cls_dim_val(DS_model,
                     cls_model,
                     dim_model,
                     baseBERT,
                     dloader,
                     dloader_val,
                     checkpoint_file,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     mix_ratio=1,
                     lr=1e-4,
                     epoch=100,
                     log_interval=10,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     use_pi=False):
    global alpha, beta
    
    DS_model.to(device)
    cls_model.to(device)
    baseBERT.to(device)
    dim_model.to(device)
        
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_cls = optim.Adam(cls_model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        cls_model = torch.nn.DataParallel(cls_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
        dim_model = torch.nn.DataParallel(dim_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    aucs = []
    best_auc = 0

    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            # if batch_idx > int(len(dloader)/5):
            #     break
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            model_optimizer_cls.zero_grad()
            model_optimizer_dim.zero_grad()
            
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
                       
            ptloss = True if batch_idx%699==3 else False
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            aug2 = 2*random.random()
            output2,EDisease2, (s,input_emb2,input_emb_org2), _,_, _, _ = DS_model(baseBERT,sample,noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=False)       
            
            bs = len(s)            

            loss_dim = dim_model(output[:,:1],
                             input_emb_org,
                             CLS_emb_emb,
                             nohx,
                             mask_ratio=mask_ratio,
                             mode=mode,
                             ptloss=ptloss,
                             DS_model=DS_model,
                             mix_ratio=mix_ratio,
                             EDisease2=output2[:,:1],
                             shuffle=True,
                             use_pi=use_pi,
                             yespi=yespi,
                             ep=ep
                            )

            hidden = output
            cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])

            trg_cls=make_trg(sample, train_cls=True)                
            trg_triage=(trg_cls['trg_triage']).to(DS_model.device)
            trg_icu7=(trg_cls['trg_icu7']).to(DS_model.device)
            trg_die7=(trg_cls['trg_die7']).to(DS_model.device)
            trg_poor7=(trg_cls['trg_icudeath7']).to(DS_model.device)
            
            cls_poor2 = torch.max(cls_icu,cls_die)

            loss_icu = criterion(cls_icu,trg_icu7)
            loss_die = criterion(cls_die,trg_die7)
            loss_tri = criterion(cls_tri,trg_triage)
            loss_poor = criterion(cls_poor,trg_poor7)
            
            loss = loss_poor + loss_icu + loss_die + loss_dim/(alpha+beta+1e-6) 

            if str(loss.item())=='nan':
                # debugnan.append([h.cpu(),hm.cpu(),h_emb.cpu(),h_emb_mean.cpu(),h_emb_emb.cpu()])
                debug_writer(debugnan,tp=2)                       
            else:
                loss.sum().backward()
                model_optimizer_cls.step()
                model_optimizer.step()
                model_optimizer_dim.step()
                    
                with torch.no_grad():
                    epoch_loss += loss.item()*bs
                    epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} Li:{:.4f} Ld:{:.4f} Lp:{:.4f} Ldim:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item(), loss_icu.item(), loss_die.item(), loss_poor.item(), loss_dim.item(),))
                
            if iteration % 200 == 0:
                try:
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_CLS.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='CLS.pth',
                                    model=cls_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)                    
                except: 
                    print('** error save checkpoint **')
                    pass
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_CLS.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='CLS.pth',
                            model=cls_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)
            
            val_model = AIED_bert.ewed_Model(config=config,
                                 tokanizer=BERT_tokenizer,
                                 device=device)
            
            val_DS_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',val_model)
            
            _, auc = count_tsne_cls(net_F=val_DS_model,
                               baseBERT=baseBERT,
                               cls_model=cls_model,
                               target=dloader_val,
                               device=device,
                               use_pi=use_pi,
                               ki=ep,
                               draw=False)
            
            aucs.append([ep,auc])
            
            pd_aucs = pd.DataFrame(aucs)
            pd_aucs.to_csv('./loss_record/aucs.csv', sep = ',')
            
            cls_model.train()
            if auc > best_auc:
                # cls_model.train()
                save_checkpoint(checkpoint_file=checkpoint_file,
                                checkpoint_path='EDisease_CLS_BEST.pth',
                                model=DS_model,
                                parallel=parallel)
                save_checkpoint(checkpoint_file=checkpoint_file,
                                checkpoint_path='CLS_BEST.pth',
                                model=cls_model,
                                parallel=parallel)                
                save_checkpoint(checkpoint_file=checkpoint_file,
                                checkpoint_path='DIM_BEST.pth',
                                model=dim_model,
                                parallel=parallel)                
                best_auc = auc

            print(f'======= epoch:{ep} ======== ** best auc = {best_auc}')
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune_CLS.csv', sep = ',')
    print(total_loss)  


def make_pickle(DS_model,
                baseBERT,
                dloader,
                dataset,
                sl=0,
                noise_scale=0.002,
                mask_ratio=0.33,
                lr=1e-5,
                epoch=100,
                log_interval=10,
                parallel=parallel,
                val=None):
    global checkpoint_file
    DS_model.to(device)
    baseBERT.to(device)
    baseBERT.eval()
    DS_model.eval()
    
    if device == 'cuda':
            torch.cuda.set_device(0)

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
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = DS_model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            sidx = sample['idx']

            for j, idx in enumerate(sidx):
                dsidx= dataset[dataset['idx']==idx.item()].index[0]
                dataset.at[dsidx,'ccemb'] = c_emb[j].cpu()
                dataset.at[dsidx,'hxemb'] = h_emb_mean[j].cpu()
                # dataset.at[dsidx,'piemb'] = p_emb[j].cpu()

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        0))  
    if val is None:
        pklfile = 'nhamcs.pickle'
    else:
        pklfile = 'nhamcs_val.pickle'

    try:
        with open(pklfile,'wb') as f:
            pickle.dump(dataset,f)
            print('** complete create pickle **')
    except:
        pass 
    return dataset  
        
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
            
config = {'hidden_size': 96,
          'bert_hidden_size': 768,
          'max_position_embeddings':512,
          'eps': 1e-12,
          'input_size': 64,
          'vocab_size':64,
          'type_vocab_size':4,
          'hidden_dropout_prob': 0.1,
          'num_attention_heads': 12, 
          'attention_probs_dropout_prob': 0.2,
          'intermediate_size': 64,
          'num_hidden_layers': 12,
          'structure_size':15,
          'order_size':256
         }

pretrained_weights="bert-base-multilingual-cased"
BERT_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
emb_model = AIED_bert.ewed_Model(config=config,
                                 tokanizer=BERT_tokenizer,
                                 device=device)
pickle_Model = AIED_bert.pickle_Model(config=config,
                                 tokanizer=BERT_tokenizer,
                                 device=device)

dim_model = AIED_bert.DIM(config=config,
                          device=device,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma)

expand_model = AIED_bert.ewed_expand_Model(config=config)

cls_model = AIED_bert.ewed_CLS_Model(config=config,device=device)

print('emb_model PARAMETERS: ' ,AIED_bert.count_parameters(emb_model))
print('dim_model PARAMETERS: ' ,AIED_bert.count_parameters(dim_model))
print('cls_model PARAMETERS: ' ,AIED_bert.count_parameters(cls_model))
print('pickle_Model PARAMETERS: ' ,AIED_bert.count_parameters(pickle_Model))

pretrained_weights="bert-base-multilingual-cased"
BERT_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

baseBERT = AIED_bert.bert_baseModel()
print(' ** confirm pretrained BERT from_pretrained BERT ** ')

try: 
    baseBERT = load_checkpoint(checkpoint_file,'BERT_ml_pretrain4.pth',baseBERT)
    print(' ** Complete Load baseBERT Model ** ')
except:
    print('*** No Pretrain_baseBERT_Model ***')

for param in baseBERT.parameters():
    param.requires_grad = False   
print(' ** pretrained BERT WEIGHT ** ')
print('baseBERT PARAMETERS: ' ,AIED_bert.count_parameters(baseBERT))

# baseBERT = AIED_bert.bert_baseModel()
# print(' ** confirm pretrained BERT from_pretrained BERT ** ')
# for param in baseBERT.parameters():
#     param.requires_grad = False   
# print(' ** FIX pretrained BERT WEIGHT ** ')
# save_checkpoint(checkpoint_file=checkpoint_file,
#                 checkpoint_path='BERT_ml.pth',
#                 model=baseBERT,
#                 parallel=False)
# print('baseBERT PARAMETERS: ' ,AIED_bert.count_parameters(baseBERT))

success_load = 0

if task=='train':
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
    print('use_pi = ', use_pi)

    train_AIemb(DS_model=emb_model,
                dim_model=dim_model,
                baseBERT=baseBERT,
                dloader=EDEW_DL,
                normalization=structure_std,                
                noise_scale=0.001,
                mask_ratio=0.33,
                lr=1e-5,
                epoch=100,
                log_interval=15,
                parallel=parallel,
                sl=success_load)

elif task=='pickle':
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
        print(' ** Complete Load EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_Model ***')
        emb_model = pickle_Model
        success_load = 1

    try:     
        dim_model = load_checkpoint(checkpoint_file,'DIM.pth',dim_model)
    except:
        print('*** No Pretrain_DIM_Model ***')
        pass
    print('** Start load pickle **')
    
    pklfile = 'trainset_pickle_pi_evm_pBERTd_3.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
    
    data15_triage_train = data15_triage_train.reset_index() 
    
    structural = ['Age',
                  'Sex',
                  'ACCOUNTSEQNO',
                  'SYSTOLIC', 
                  'DIASTOLIC',
                  'PULSE',
                  'OXYGEN',
                  'RESPIRATION',
                  'BODYTEMPERATURE',
                  'HEIGHT', 
                  'WEIGHT',
                  'PAININDEX',
                  'BE',
                  'BV',
                  'BM']
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T                
    
    EDEW_DS = AIED_dataloader.pickle_Dataset(ds=data15_triage_train,
                                             normalization = dm_normalization_np,
                                             use_pi=use_pi
                                          )

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                             shuffle = True,
                             num_workers = nw,
                             batch_size = batch_size)

    structure_mean = dm_normalization_np[0]
    structure_std  = dm_normalization_np[1]
    
    print('dm_mean', structure_mean)
    print('dm_std', structure_std)    
    
    print('batch_size = ', batch_size)
    print('use_pi = ', use_pi)

    train_pickle(DS_model=emb_model,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 dloader=EDEW_DL,
                 noise_scale=0.1,
                 mask_ratio=0.33,
                 lr=1e-4,
                 epoch=100000,
                 log_interval=15,
                 parallel=parallel,
                 sl=success_load,
                 mix_ratio=1,
                 use_pi=use_pi)   
    
elif task=='test':
    print('use_pi = ', use_pi)
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
        print(' ** Complete Load EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_Model ***')
        success_load = 1
    
    all_datas = AIED_dataloader.load_datas()

    data15_triage_val = all_datas['data15_triage_val']
    data01_person = all_datas['data01_person']
    data02_wh = all_datas['data02_wh']
    data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']  
    data07_death = all_datas['data07_death']
    data25_icu = all_datas['data25_icu']
    data15_triage_train = all_datas['data15_triage_train']
    
    data15_triage_val_sample = data15_triage_val.sample(frac=0.25,random_state=9)
    data15_triage_train_sample = data15_triage_train.sample(frac=0.01,random_state=9)
    
    data15_triage_val_sample = data15_triage_train.reset_index()
    
    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds=data15_triage_val_sample,
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
                             shuffle = False,
                             num_workers=3,
                             batch_size=8,
                             collate_fn=AIED_dataloader.collate_fn)
    
    f_edisease = test_AIemb(DS_model=emb_model,
                            baseBERT=baseBERT,
                            dloader=EDEW_DL,
                            parallel=parallel,
                            device=device,
                            use_pi=use_pi,
                            tsne=False)
    
    print('** Create pickle **')
    pklfile = './pickle_query_result/trainset_query_result_pickle.pickle'
    with open(pklfile,'wb') as f:
        pickle.dump([f_edisease,EDEW_DS,data15_triage_val_sample],f)
        print('** complete create pickle **')    
        
    print(' ** draw 2d **')
    AIED_utils.plot_2d(f_edisease)

elif task=='test_cls':
    print('use_pi = ', use_pi)
    checkpoint_file = './checkpoint_emb/cls_pBERT4_evmpi_wo'
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',emb_model)
        print(' ** Complete Load EDisease_CLS Model ** ')
    except:
        print('*** No Pretrain_EDisease_Model ***')
        success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass        
        
    all_datas = AIED_dataloader.load_datas()

    data15_triage_val = all_datas['data15_triage_val']
    data01_person = all_datas['data01_person']
    data02_wh = all_datas['data02_wh']
    data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']  
    data07_death = all_datas['data07_death']
    data25_icu = all_datas['data25_icu']
    data15_triage_train = all_datas['data15_triage_train']
    
    data15_triage_val_sample = data15_triage_val.sample(frac=0.25,random_state=9)
    data15_triage_train_sample = data15_triage_train.sample(frac=0.01,random_state=9)
    
    data15_triage_val_sample = data15_triage_val_sample.reset_index()
    
    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds=data15_triage_val_sample,
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
                             shuffle = False,
                             num_workers=4,
                             batch_size=32,
                             collate_fn=AIED_dataloader.collate_fn)
    
    f_edisease = test_AIcls(DS_model=emb_model,
                            baseBERT=baseBERT,
                            cls_model=cls_model,
                            dloader=EDEW_DL,
                            parallel=parallel,
                            device=device,
                            use_pi=use_pi,
                            tsne=True)

    print('** Create pickle **')
    pklfile = './pickle_cls_result/valset_cls_result_pickle.pickle'
    with open(pklfile,'wb') as f:
        pickle.dump(f_edisease,f)
        print('** complete create pickle **')    
    
    print(' ** draw 2d **')
    AIED_utils.plot_2d(f_edisease)    
    
elif task=='query':    
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
        print(' ** Complete Load EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_Model ***')
        success_load = 1

    all_datas = AIED_dataloader.load_datas()

    data15_triage_val = all_datas['data15_triage_val']
    data01_person = all_datas['data01_person']
    data02_wh = all_datas['data02_wh']
    data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']  
    data07_death = all_datas['data07_death']
    data25_icu = all_datas['data25_icu']
    data15_triage_train = all_datas['data15_triage_train']
    
    data15_triage_val_sample = data15_triage_val.sample(frac=0.1,random_state=9)
    data15_triage_train_sample = data15_triage_train.sample(frac=0.01,random_state=9)
    
    query_sample = None
    
    if query_sample is not None:
        query_sample['query'] = 1
        query_sample['idx'] = -1-1*np.arange(len(query_sample))
        data15_triage_val_sample['query'] = 0
        data15_triage_val_sample['idx'] = np.arange(len(data15_triage_val_sample))
        query_val_sample = pd.concat([query_sample,data15_triage_val_sample])
        query_val_sample = query_val_sample.reset_index()
    else:
        query_val_sample = data15_triage_val_sample
        query_val_sample = query_val_sample.reset_index()
    
    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds=query_val_sample,
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
                             shuffle = False,
                             num_workers=28,
                             batch_size=4,
                             collate_fn=AIED_dataloader.collate_fn)
    
    f_edisease, idx = test_AIemb(DS_model=emb_model,
               dloader=EDEW_DL,
               parallel=parallel,
               device=device)
    if query_sample is not None:
        from AIED_utils import draw_predict
        query_edisease = f_edisease[:len(query_sample)]
        val_edisease = f_edisease[len(query_sample):]

        ars = AIED_utils.query(query_edisease[0],val_edisease)
        k_ars = ars[0:6]
        
        idxx = idx[k_ars]
        
        for i in idxx:
            idxs = query_val_sample[query_val_sample['idx']==i].index[0]                                                    
            print(i,idxs)
            sample = EDEW_DS[idxs]
            print(sample['trg'][-5:-3])
            print(sample['portal'])

elif task=='pickle_cls':
    
    if use_pi:
        checkpoint_file = './checkpoint_emb/cls_v601_wvmpi_w'
    else:
        checkpoint_file = './checkpoint_emb/cls_pBERT2_evmpi_wo'
        
    batch_size = 256
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
            success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass
    
    print('** Start load pickle **')
    
    pklfile = 'trainset_pickle_pi_evm_pBERTd_3.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
    
    data15_triage_train = data15_triage_train.reset_index()
        
    structural = ['Age',
                  'Sex',
                  'ACCOUNTSEQNO',
                  'SYSTOLIC', 
                  'DIASTOLIC',
                  'PULSE',
                  'OXYGEN',
                  'RESPIRATION',
                  'BODYTEMPERATURE',
                  'HEIGHT', 
                  'WEIGHT',
                  'PAININDEX',
                  'BE',
                  'BV',
                  'BM']
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T                
        
    rn0 = data15_triage_train['TRIAGE']>2
    rp0 = data15_triage_train['TRIAGE']<3

    rn1 = data15_triage_train['icu7']>7
    rp1 = data15_triage_train['icu7']<8

    rn2 = data15_triage_train['death7']>7
    rp2 = data15_triage_train['death7']<8    

    g000 = pd.Series(data15_triage_train[rn0 & rn1 & rn2].index)
    g001 = pd.Series(data15_triage_train[rn0 & rn1 & rp2].index)
    g010 = pd.Series(data15_triage_train[rn0 & rp1 & rn2].index)
    g011 = pd.Series(data15_triage_train[rn0 & rp1 & rp2].index)
    g100 = pd.Series(data15_triage_train[rp0 & rn1 & rn2].index)
    g101 = pd.Series(data15_triage_train[rp0 & rn1 & rp2].index)
    g110 = pd.Series(data15_triage_train[rp0 & rp1 & rn2].index)
    g111 = pd.Series(data15_triage_train[rp0 & rp1 & rp2].index)
    
    ltemp = [g000] + [g001]*290 + [g010]*339 + [g011]*6238 + [g100]*4 + [g101]*200 + [g110]*475 + [g111]*7988
    print(len(g000),len(g001),len(g010),len(g011),len(g100),len(g101),len(g110),len(g111))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')
    
    EDEW_DS = AIED_dataloader.pickle_Dataset(ds=data15_triage_train,
                                             normalization = dm_normalization_np,
                                             dsidx=dtemp,
                                             use_pi=use_pi
                                            )

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                             shuffle = True,
                             num_workers = nw,
                             batch_size = batch_size)

    structure_mean = dm_normalization_np[0]
    structure_std  = dm_normalization_np[1]
    
    print('dm_mean', structure_mean)
    print('dm_std', structure_std)    
    
    print('batch_size = ', batch_size)
    print('use_pi = ', use_pi)

    train_pickle_cls(DS_model=emb_model,
                     cls_model=cls_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=1e-4,
                     epoch=10000,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi)            

    
elif task=='nhamcs_cls':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',emb_model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']  
    datas_all = all_datas['datas']

    structure_mean = dm_normalization_np[0]
    structure_std  = dm_normalization_np[1]
    
    
    data15_triage_train = datas_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    
    ltemp = [g00] + [g01]*int(len(g00)/len(g01)) +[g10]*int(len(g00)/len(g10)) +[g11]*int(len(g00)/len(g11))
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=AIED_dataloader.collate_fn)

    train_NHAMCS_cls(DS_model=emb_model,
                     cls_model=cls_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-5,
                     epoch=1001,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi)   

elif task=='pickle_nhamcs_cls':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
            success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  
    
    
    # data15_triage_train = data15_triage_train[data15_triage_train['AGE']>=18]
    
    data15_triage_train = data15_triage_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    # g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    # g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    # g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    # g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    # ltemp = [g00] + [g01]*int(len(g00)/len(g01)) +[g10]*int(len(g00)/len(g10)) +[g11]*int(len(g00)/len(g11))
    # print(len(g00),len(g01),len(g10),len(g11))
    # print(' *** balance the dataset p/n *** ')
    # dtemp = pd.concat(ltemp)  
    # print(' *** balance complete *** ')

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)) +[g11]*300)*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls(DS_model=emb_model,
                     cls_model=cls_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-4,
                     epoch=20,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi) 

elif task=='pickle_nhamcs_cls_dim':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
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

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  
    
    
    # data15_triage_train = data15_triage_train[data15_triage_train['AGE']>=18]
    
    data15_triage_train = data15_triage_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    # g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    # g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    # g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    # g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    # ltemp = [g00] + [g01]*int(len(g00)/len(g01)) +[g10]*int(len(g00)/len(g10)) +[g11]*int(len(g00)/len(g11))
    # print(len(g00),len(g01),len(g10),len(g11))
    # print(' *** balance the dataset p/n *** ')
    # dtemp = pd.concat(ltemp)  
    # print(' *** balance complete *** ')

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)) +[g11]*300)*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls_dim(DS_model=emb_model,
                     cls_model=cls_model,
                     dim_model=dim_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-5,
                     epoch=100001,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi) 
    
elif task=='pickle_nhamcs':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
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

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  
    
    
    data15_triage_train = data15_triage_train.reset_index()
                                

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS(DS_model=emb_model,
                     dim_model=dim_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,                 
                     mix_ratio=1,
                     lr=1e-3,
                     epoch=301,
                     log_interval=15,
                     parallel=parallel,
                     checkpoint_file=checkpoint_file,
                     use_pi=False)      

elif task=='test_nhamcs_cls':
    batch_size = 16
    use_pi= False
    parallel = False
    # try: 
    #     emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',emb_model)
    #     print(' ** Complete Load CLS EDisease Model ** ')
    # except:
    #     print('*** No Pretrain_EDisease_CLS_Model ***')

    #     try: 
    #         emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
    #         print(' ** Complete Load EDisease Model ** ')
    #     except:
    #         print('*** No Pretrain_EDisease_Model ***')
    #         success_load = 1

    # try:     
    #     cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    # except:
    #     print('*** No Pretrain_cls_Model ***')
    #     pass

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']  
    datas_all = all_datas['datas']
    
    # datas_val = datas_val[datas_val['AGE']>=18]
    
    data15_triage_val_sample = datas_test.reset_index()

    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
    import AIED_utils    
    # checkpoint_file = './checkpoint_emb/Revision_dc'    
    ki = 'BEST'
    
    try: 
        emb_model = load_checkpoint(checkpoint_file,f'EDisease_CLS_{ki}.pth',emb_model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,f'CLS_{ki}.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass
    f_edisease = test_AIcls(DS_model=emb_model,
                            baseBERT=baseBERT,
                            cls_model=cls_model,
                            dloader=EDEW_DL,
                            parallel=parallel,
                            device=device,
                            use_pi=use_pi,
                            tsne=False)

elif task=='nhamcs':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'DIM.pth',dim_model)
    except:
        print('*** No Pretrain_DIM_Model ***')
        pass

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']   
    datas_all = all_datas['datas']

    structure_mean = dm_normalization_np[0]
    structure_std  = dm_normalization_np[1]
    
    
    data15_triage_train = datas_train.reset_index()
                             
    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=AIED_dataloader.collate_fn)

    train_NHAMCS(DS_model=emb_model,
                     dim_model=dim_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-5,
                     epoch=1001,
                     log_interval=15,
                     parallel=parallel,
                     checkpoint_file=checkpoint_file,
                     use_pi=False)   
    
elif task == 'make_pickle':
    batch_size = 16
    use_pi= False
    parallel = False

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']    
    datas_all = all_datas['datas']
    
    data15_triage_val_sample = datas_train

    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
          
    data15_triage_val_sample['ccemb']=None
    data15_triage_val_sample['hxemb']=None
    data15_triage_val_sample['piemb']=None

    dataset = make_pickle(DS_model=emb_model,
                          baseBERT=baseBERT,
                          dloader=EDEW_DL,
                          dataset=data15_triage_val_sample)    

elif task == 'make_pickle_val':
    batch_size = 16
    use_pi= False
    parallel = False

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']     
    datas_all = all_datas['datas']
    
    data15_triage_val_sample = datas_val

    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
          
    data15_triage_val_sample['ccemb']=None
    data15_triage_val_sample['hxemb']=None
    data15_triage_val_sample['piemb']=None

    dataset = make_pickle(DS_model=emb_model,
                          baseBERT=baseBERT,
                          dloader=EDEW_DL,
                          dataset=data15_triage_val_sample,
                          val='val')   

elif task=='pickle_nhamcs_val_cls':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
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

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  

    
    print('** Start load val pickle **')
    
    pklfile = 'nhamcs_val.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0    
    
    data15_triage_train = data15_triage_train.sample(frac=0.33612,random_state=9).reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)))*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls(DS_model=emb_model,
                     cls_model=cls_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,           
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-4,
                     epoch=45,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi) 
    
elif task=='count_nhamcs_cls':
    batch_size = 16
    use_pi= False
    parallel = False

    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']  
    datas_all = all_datas['datas']
    
    # datas_val = datas_val[datas_val['AGE']>=18]
    
    data15_triage_val_sample = datas_test.reset_index()

    EDEW_DS = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
    import AIED_utils    
    
    res = []
    for ki in range(20):
    # ki = 0
        try: 
            emb_model = load_checkpoint(checkpoint_file,f'EDisease_CLS_{ki}.pth',emb_model)
            print(' ** Complete Load CLS EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_CLS_Model ***')
    
            try: 
                emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',emb_model)
                print(' ** Complete Load EDisease Model ** ')
            except:
                print('*** No Pretrain_EDisease_Model ***')
                success_load = 1
    
        try:     
            cls_model = load_checkpoint(checkpoint_file,f'CLS_{ki}.pth',cls_model)
        except:
            print('*** No Pretrain_cls_Model ***')
            pass
        f_edisease, auc = count_AIcls(DS_model=emb_model,
                                baseBERT=baseBERT,
                                cls_model=cls_model,
                                dloader=EDEW_DL,
                                parallel=parallel,
                                device=device,
                                use_pi=use_pi,
                                ki=ki,
                                draw=True)
        
        print(ki,'auc= ',auc)
        res.append([ki,auc])


elif task=='pickle_nhamcs_cls_val':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
            success_load = 1

    try:     
        cls_model = load_checkpoint(checkpoint_file,'CLS.pth',cls_model)
    except:
        print('*** No Pretrain_cls_Model ***')
        pass

# ====
    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']  
    datas_all = all_datas['datas']
    
    # datas_val = datas_val[datas_val['AGE']>=18]
    
    data15_triage_val_sample = datas_val.reset_index()

    EDEW_DS_val = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL_val = DataLoader(dataset = EDEW_DS_val,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
# ====

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
          
    data15_triage_train = data15_triage_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)) +[g11]*300)*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls_val(DS_model=emb_model,
                     cls_model=cls_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,
                     dloader_val=EDEW_DL_val,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-4,
                     epoch=300,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi)  
    
elif task=='pickle_nhamcs_cls_dim_val':
    batch_size = 1024
    use_pi= False
    parallel = False
    try: 
        emb_model = load_checkpoint(checkpoint_file,'EDisease_CLS.pth',pickle_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

        try: 
            emb_model = load_checkpoint(checkpoint_file,'EDisease.pth',pickle_Model)
            print(' ** Complete Load EDisease Model ** ')
        except:
            print('*** No Pretrain_EDisease_Model ***')
            emb_model = pickle_Model
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

# ====
    all_datas = AIED_dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']   
    datas_all = all_datas['datas']
    
    # datas_val = datas_val[datas_val['AGE']>=18]
    
    data15_triage_val_sample = datas_val.reset_index()

    EDEW_DS_val = AIED_dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL_val = DataLoader(dataset = EDEW_DS_val,
                         shuffle = False,
                         num_workers=4,
                         batch_size=128,
                         collate_fn=AIED_dataloader.collate_fn)
# ====

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  
    
    
    # data15_triage_train = data15_triage_train[data15_triage_train['AGE']>=18]
    
    data15_triage_train = data15_triage_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    # g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    # g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    # g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    # g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    # ltemp = [g00] + [g01]*int(len(g00)/len(g01)) +[g10]*int(len(g00)/len(g10)) +[g11]*int(len(g00)/len(g11))
    # print(len(g00),len(g01),len(g10),len(g11))
    # print(' *** balance the dataset p/n *** ')
    # dtemp = pd.concat(ltemp)  
    # print(' *** balance complete *** ')

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)) +[g11]*300)*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls_dim_val(DS_model=emb_model,
                     cls_model=cls_model,
                     dim_model=dim_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,
                     dloader_val=EDEW_DL_val,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-5,
                     epoch=100001,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi) 