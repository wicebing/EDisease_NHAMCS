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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def save_checkpoint(checkpoint_file,checkpoint_path, model, parallel, optimizer=None):
    if parallel:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            name = k[7:] # remove module.
            state_dict[name] = v
    else:
        state_dict = model.state_dict()

    state = {'state_dict': state_dict,}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(checkpoint_file,checkpoint_path))

    print('model saved to %s / %s' % (checkpoint_file,checkpoint_path))
    
def load_checkpoint(checkpoint_file,checkpoint_path, model):
    state = torch.load(os.path.join(checkpoint_file,checkpoint_path),
                       map_location='cuda:0'
#                       map_location={'cuda:0':'cuda:1'}
                       )
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s / %s' % (checkpoint_file,checkpoint_path))
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join(checkpoint_file,name_))
    print('model saved to %s / %s' % (checkpoint_file,name_))
    return model

def draw_tsne(net_F, baseBERT,target, device):
    from sklearn import manifold
    net_F.to(device)      
    net_F.eval()
    
    f_target_ = []
    triage_tlabel_ = []
    hxnum_label_ = []
    with torch.no_grad():
        t0 = time.time()
        for batch_idx, sample in enumerate(target):
            s_np = sample['structure'].numpy()
            c_np = sample['cc'].numpy()
            h_np = sample['ehx'].numpy()
            nans = np.all(np.isnan(s_np))
            nanc = np.all(np.isnan(c_np))
            nanh = np.all(np.isnan(h_np))
            
            if ~nans and ~nanc and ~nanh:
                output, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(h,hm,h_emb,h_emb_mean,h_emb_emb), nohx, expand_data = net_F(baseBERT,sample)

                bs = len(s)
                # EDisease = output[:,:1]
                err = torch.all(torch.isnan(output[:,0]))
                if err:    
                    print('EDisease has Nan')
                else:
                    f_target_.append(output[:,0].cpu())
                    triage_tlabel_.append(s[:,-1].view(-1).cpu())
                    hxnum_label_.append((nohx > 1).int().view(-1).cpu())                    
               
                if batch_idx%100 ==0:
                    print('[{} ({:.0f}%)/ ep_time:{:.0f}min]'.format(batch_idx,
                            100. * batch_idx / len(target),
                            (time.time()-t0)*len(target)/(60*(batch_idx+1))))  
            else:
                print('nans: ', nans)
                print('nanc: ', nanc)
                print('nanh: ', nanh)
    
    f_target = torch.cat(f_target_,dim=0)
    
    triage_tlabel = torch.cat(triage_tlabel_,dim=0)
    hxnum_label = torch.cat(hxnum_label_,dim=0)
    f_tlabel = [0]*len(f_target)
    print(f_target.shape)
    
    labels = {'f_tlabel':f_tlabel,
              'triage_tlabel':triage_tlabel.view(-1).cpu().numpy(),
              'hxnum_label':hxnum_label.view(-1).cpu().numpy()}
    
    #draw tsne
    print(' ** tsne ** ')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    T_tsne = tsne.fit_transform(f_target.numpy())
    print(' ** draw tsne ** ')
    plot_1(T_tsne,labels)
#    return T_tsne, S_tsne, f_tlabel, f_slabel

def draw_tsne_v2(net_F,baseBERT, target, device,use_pi,tsne=True):
    from sklearn import manifold
    net_F.to(device)      
    net_F.eval()
    baseBERT.to(device)
    baseBERT.eval()
    
    f_target_ = []
    triage_tlabel_ = []
    hxnum_label_ = []
    all_label_ = []
    idx_ =[]
    
    with torch.no_grad():
        t0 = time.time()
        for batch_idx, sample in enumerate(target):
            s_np = sample['structure'].numpy()
            c_np = sample['cc'].numpy()
            h_np = sample['ehx'].numpy()
            nans = np.all(np.isnan(s_np))
            nanc = np.all(np.isnan(c_np))
            nanh = np.all(np.isnan(h_np))
            
            trg = sample['trg']
            
            if ~nans and ~nanc and ~nanh:
                output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),_ , nohx, expand_data = net_F(baseBERT,sample,use_pi=use_pi,mask_ratio=0,mask_ratio_pi=0)

                bs = len(s)
                # EDisease = output[:,:1]
                err = torch.all(torch.isnan(output[:,0]))
                if err:    
                    print('EDisease has Nan')
                else:
                    hidden = output
                    f_target_.append(hidden[:,0].cpu())
                    #print(output[:,0,:12].cpu())
                    triage_tlabel_.append(s[:,-1].view(-1).cpu())
                    hxnum_label_.append((nohx > 1).int().view(-1).cpu())
                    all_label_.append(trg.cpu())
                    idx_.append(sample['idx'])
               
                if batch_idx%100 ==0:
                    print('[{} ({:.0f}%)/ ep_time:{:.0f}min]'.format(batch_idx,
                            100. * batch_idx / len(target),
                            (time.time()-t0)*len(target)/(60*(batch_idx+1))))  
            else:
                print('nans: ', nans)
                print('nanc: ', nanc)
                print('nanh: ', nanh)
    
    f_target = torch.cat(f_target_,dim=0)
    
    triage_tlabel = torch.cat(triage_tlabel_,dim=0)
    hxnum_label = torch.cat(hxnum_label_,dim=0)
    f_tlabel = [0]*len(f_target)
    all_label = torch.cat(all_label_,dim=0)
    idx = torch.cat(idx_,dim=0)
    print(f_target.shape)
    
    labels = {'f_tlabel':f_tlabel,
              'triage_tlabel':triage_tlabel.view(-1).cpu().numpy(),
              'hxnum_label':hxnum_label.view(-1).cpu().numpy(),
              'all_label':all_label.numpy()}
    
    if tsne:
    #draw tsne
        print(' ** tsne ** ')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        T_tsne = tsne.fit_transform(f_target.numpy())
        print(' ** draw tsne ** ')
        plot_1_v2(T_tsne,labels)
    
    result ={'target':f_target.numpy(),
             'idx':idx.cpu().numpy(),
             'label':labels}
    return result

def draw_tsne_cls(net_F,baseBERT,cls_model, target, device,use_pi,tsne=True):
    from sklearn import manifold

    net_F.to(device)      
    net_F.eval()
    baseBERT.to(device)
    baseBERT.eval()
    cls_model.to(device)
    cls_model.eval()
    
    f_target_ = []
    triage_tlabel_ = []
    hxnum_label_ = []
    all_label_ = []
    idx_ =[]
    
    icu_ = []
    die_ = []
    tri_ = []
    poor_ = []
    poor2_ = []
    
    with torch.no_grad():
        t0 = time.time()
        for batch_idx, sample in enumerate(target):
            s_np = sample['structure'].numpy()
            c_np = sample['cc'].numpy()
            h_np = sample['ehx'].numpy()
            nans = np.all(np.isnan(s_np))
            nanc = np.all(np.isnan(c_np))
            nanh = np.all(np.isnan(h_np))
            
            trg = sample['trg']
            
            if ~nans and ~nanc and ~nanh:
                output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),_ , nohx, expand_data = net_F(baseBERT,sample,use_pi=use_pi,test=True)

                bs = len(s)
                # EDisease = output[:,:1]
                #using the hidden layer as EDisease, output for pretrain
                hidden = output
                f_target_.append(hidden[:,0].cpu())
                #print(output[:,0,:12].cpu())
                triage_tlabel_.append(s[:,-1].view(-1).cpu())
                hxnum_label_.append((nohx > 1).int().view(-1).cpu())
                all_label_.append(trg.cpu())
                idx_.append(sample['idx'])

                cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])
                
                cls_icu_soft = nn.Softmax(dim=-1)(cls_icu)
                cls_die_soft = nn.Softmax(dim=-1)(cls_die)
                cls_tri_soft = nn.Softmax(dim=-1)(cls_tri)
                cls_poor_soft = nn.Softmax(dim=-1)(cls_poor)
                               
                icu_.append(cls_icu_soft[:,-1].view(-1).cpu())
                die_.append(cls_die_soft[:,-1].view(-1).cpu())
                tri_.append(cls_tri_soft[:,-1].view(-1).cpu())
                poor_.append(cls_poor_soft[:,-1].view(-1).cpu())
                
                cls_poor2 = torch.max(cls_icu,cls_die)
                cls_poor_soft2 = nn.Softmax(dim=-1)(cls_poor2)
                poor2_.append(cls_poor_soft2[:,-1].view(-1).cpu())
                
                if batch_idx%100 ==0:
                    print('[{} ({:.0f}%)/ ep_time:{:.0f}min]'.format(batch_idx,
                            100. * batch_idx / len(target),
                            (time.time()-t0)*len(target)/(60*(batch_idx+1))))  
            else:
                print('nans: ', nans)
                print('nanc: ', nanc)
                print('nanh: ', nanh)
    
    f_target = torch.cat(f_target_,dim=0)
    
    icu = torch.cat(icu_,dim=0)
    die = torch.cat(die_,dim=0)
    tri = torch.cat(tri_,dim=0)
    poor = torch.cat(poor_,dim=0)
      
    triage_tlabel = torch.cat(triage_tlabel_,dim=0)
    hxnum_label = torch.cat(hxnum_label_,dim=0)
    f_tlabel = [0]*len(f_target)
    all_label = torch.cat(all_label_,dim=0)
    idx = torch.cat(idx_,dim=0)
    print(f_target.shape)
    
    labels = {'f_tlabel':f_tlabel,
              'triage_tlabel':triage_tlabel.view(-1).cpu().numpy(),
              'hxnum_label':hxnum_label.view(-1).cpu().numpy(),
              'all_label':all_label.numpy()}
    
    pred = {'icu':icu,
            'die':die,
            'tri':tri,
            'poor':poor
           }
    
    if tsne:
        #draw tsne
        print(' ** tsne ** ')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        T_tsne = tsne.fit_transform(f_target.numpy())
        print(' ** draw tsne ** ')
        plot_1_v2(T_tsne,labels)
        plot_1_v_nhamcs(T_tsne,labels)
    
    print(' ** draw ROC AUC ** ')
    plot_roc(pred,labels)
    plot_roc_nhamcs2(pred,labels)
    
    result ={'target':f_target.numpy(),
             'idx':idx.cpu().numpy(),
             'label':labels,
             'pred':pred}
    return result

def plot_roc(pred,labels,ep=0):
    from sklearn.metrics import roc_curve, auc
    ROC_threshold = torch.linspace(0,1,100).numpy()
    
    all_label = labels['all_label']
    
    triage_tlabel_ = all_label[:,1]    
    icu_tlabel_ = all_label[:,3]
    die_tlabel_ = all_label[:,4]

    triage_tlabel = []
    for a in triage_tlabel_:
        triage_tlabel.append(int(a<3))
    triage_tlabel = np.array(triage_tlabel)    
    
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
    
    pred_icu = pred['icu']
    pred_die = pred['die']
    pred_tri = pred['tri']
    pred_poor = pred['poor']
    
    # pred_poor = torch.max(pred_icu,torch.max(pred_die,pred_poor))
    
    rocset = ['Triage','ICU-7','Death-7','Poor-7']
    roc_label = [triage_tlabel,icu_tlabel,die_tlabel,poor_tlabel]
    roc_pred = [pred_tri,pred_icu,pred_die,pred_poor]
    
    fig = plt.figure(figsize=(20,20),dpi=200)
    
    for i, s in enumerate(rocset):
        pn = '22{:.0f}'.format(i+1)
  
        ax = fig.add_subplot(pn)
        fpr, tpr, _ = roc_curve(roc_label[i], roc_pred[i].numpy())
        roc_auc2 = auc(fpr,tpr)

        label_auc2 = str(s)+', AUROC='+str(roc_auc2)
        ax.plot(fpr,tpr,label=label_auc2)
        ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')

        ax.set_xlabel('1-specificity')
        ax.set_ylabel('sensitivity')
        ax.set_title('ROC curve without pretrain')
        plt.legend()
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.legend()
    plt.savefig('./pic_ROC/AUC_'+str(ep)+'.png')

def plot_roc_nhamcs2(pred,labels,ep=0):
    from sklearn.metrics import roc_curve, auc
    ROC_threshold = torch.linspace(0,1,100).numpy()
    
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
    
    pred_icu = pred['icu']
    pred_die = pred['die']
    pred_poor = pred['poor']
    
    rocset = ['ICU','Death','Critical']
    roc_label = [icu_tlabel,die_tlabel,poor_tlabel]
    roc_pred = [pred_icu,pred_die,pred_poor]
    
    fig = plt.figure(figsize=(36,10),dpi=200)
    
    for i, s in enumerate(rocset):
        pn = '13{:.0f}'.format(i+1)
  
        ax = fig.add_subplot(pn)
        fpr, tpr, _ = roc_curve(roc_label[i], roc_pred[i].numpy())
        roc_auc2 = auc(fpr,tpr)

        label_auc2 = str(s)+', AUROC='+str(roc_auc2)
        ax.plot(fpr,tpr,label=label_auc2)
        ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')

        ax.set_xlabel('1-specificity')
        ax.set_ylabel('sensitivity')
        ax.set_title('ROC curve without pretrain')
        plt.legend()
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.legend()
    plt.savefig('./pic_ROC/AUC_nhamcs_'+str(ep)+'.png')

    sepname = ['Adult','Child']
    for jj, sss in enumerate(sep):
        roc_label = [icu_tlabel[sss],die_tlabel[sss],poor_tlabel[sss]]
        roc_pred = [pred_icu[sss],pred_die[sss],pred_poor[sss]]
    
        fig = plt.figure(figsize=(36,10),dpi=200)
        
        for i, s in enumerate(rocset):
            pn = '13{:.0f}'.format(i+1)
      
            ax = fig.add_subplot(pn)
            fpr, tpr, _ = roc_curve(roc_label[i], roc_pred[i].numpy())
            roc_auc2 = auc(fpr,tpr)
    
            label_auc2 = str(s)+', AUROC='+str(roc_auc2)
            ax.plot(fpr,tpr,label=label_auc2)
            ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')
    
            ax.set_xlabel('1-specificity')
            ax.set_ylabel('sensitivity')
            ax.set_title('ROC curve without pretrain')
            plt.legend()
            plt.xlim(0.,1.)
            plt.ylim(0.,1.)
            plt.legend()
        plt.savefig(f'./pic_ROC/AUC_nhamcs_{sepname[jj]}_{str(ep)}.png')

def plot_roc_nhamcs(pred,labels,ep=0):
    from sklearn.metrics import roc_curve, auc
    ROC_threshold = torch.linspace(0,1,100).numpy()
    
    all_label = labels['all_label']
    
    triage_tlabel_ = all_label[:,1]    
    icu_tlabel_ = all_label[:,3]
    die_tlabel_ = all_label[:,4]
    
    age = all_label[:,5]

    # triage_tlabel = []
    # for a in triage_tlabel_:
    #     triage_tlabel.append(int(a<3))
    # triage_tlabel = np.array(triage_tlabel)    
    
    # icu_tlabel = []
    # for a in icu_tlabel_:
    #     icu_tlabel.append(int(a<8))
    # icu_tlabel = np.array(icu_tlabel)
    
    die_tlabel = []
    for a in die_tlabel_:
        die_tlabel.append(int(a<8))
    die_tlabel = np.array(die_tlabel)
    
    # poor_tlabel = []
    # for i,a in enumerate(icu_tlabel_):
    #     if icu_tlabel_[i]<8 or die_tlabel_[i]<8:
    #         poor_tlabel.append(1)
    #     else:
    #         poor_tlabel.append(0)
    # poor_tlabel = np.array(poor_tlabel)    
    
    # pred_icu = pred['icu']
    pred_die = pred['die']
    # pred_tri = pred['tri']
    # pred_poor = pred['poor']
    
    # rocset = ['Triage','ICU-7','Death-7','Poor-7']
    # roc_label = [triage_tlabel,icu_tlabel,die_tlabel,poor_tlabel]
    # roc_pred = [pred_tri,pred_icu,pred_die,pred_poor]
    
    fig = plt.figure(figsize=(20,20),dpi=200)
     
    ax = fig.add_subplot(111)
    fpr, tpr, _ = roc_curve(die_tlabel, pred_die.numpy())
    roc_auc2 = auc(fpr,tpr)

    label_auc2 = 'Death'+', AUROC='+str(roc_auc2)
    ax.plot(fpr,tpr,label=label_auc2)
    ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')

    ax.set_xlabel('1-specificity')
    ax.set_ylabel('sensitivity')
    ax.set_title('ROC curve without pretrain')
    plt.legend()
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    plt.legend()
    plt.savefig('./pic_ROC/AUC_nhamcs_'+str(ep)+'.png')
    
def plot_1_v_nhamcs(T_tsne,labels, alpha=[0.3,1]): 
    f_tlabel = labels['f_tlabel']

    hxnum_label = labels['hxnum_label']
    all_label = labels['all_label']
    
    triage_tlabel = all_label[:,1]    
    hosp_tlabel = all_label[:,2]
    icu_tlabel_ = all_label[:,3]
    die_tlabel_ = all_label[:,4]
    age_tlabel_ = all_label[:,5]
    sex_tlabel = all_label[:,6]
    cva_tlabel = all_label[:,7]
    trm_tlabel = all_label[:,8]
    query_tlabel = all_label[:,9]
    
    age_tlabel = []
    for a in age_tlabel_:
        try:
            age_tlabel.append(int(a/10))
        except:
            age_tlabel.append(-1)
    age_tlabel = np.array(age_tlabel)

    icu_tlabel = []
    for a in icu_tlabel_:
        icu_tlabel.append(int(a<8))
    icu_tlabel = np.array(icu_tlabel)
    
    die_tlabel = []
    for a in die_tlabel_:
        die_tlabel.append(int(a<8))
    die_tlabel = np.array(die_tlabel)    
    
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
                         'query']]
    '''
    
    
    fig = plt.figure(figsize=(48,10),dpi=100)
    #plt.title('EDisease (Embedding of Disease)')    
    ax = fig.add_subplot(141)
    ax.scatter(T_tsne[:,0],T_tsne[:,1],
               marker = '.', alpha=alpha[0], label = 'EDisease')
    plt.legend()
    plt.title('2D t-SNE of EDisease (Embedding of Disease)') 
    
    ax2 = fig.add_subplot(142)
    for tril in range(2):
        cls = sex_tlabel==tril
        Gender ='F' if tril==0 else 'M'
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Gender= '+str(Gender)) 
    plt.legend()
    plt.title('EDisease (Gender)') 

    ax2 = fig.add_subplot(143)
    for tril in range(12):
        cls = age_tlabel==tril
        age = str(10*tril)+' ~ '+str(10*(tril+1)-1)
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Age= '+str(age)) 
    plt.legend()
    plt.title('EDisease (Age)')     

    ax3 = fig.add_subplot(144)
    for nohx in range(2):
        cls = die_tlabel==nohx
        ax3.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[nohx], label = 'Death-7: '+str(nohx))    
    plt.legend()
    plt.title('EDisease (Death)')    
    
    #TODO sex, age, Hospital, icu, death
    
    plt.savefig('./pic/'+'EDisease_Tetra.png')
    plt.show()
    plt.close()    

def plot_1_v2(T_tsne,labels, alpha=[0.3,1]): 
    f_tlabel = labels['f_tlabel']

    hxnum_label = labels['hxnum_label']
    all_label = labels['all_label']
    
    triage_tlabel = all_label[:,1]    
    hosp_tlabel = all_label[:,2]
    icu_tlabel_ = all_label[:,3]
    die_tlabel_ = all_label[:,4]
    age_tlabel_ = all_label[:,5]
    sex_tlabel = all_label[:,6]
    cva_tlabel = all_label[:,7]
    trm_tlabel = all_label[:,8]
    query_tlabel = all_label[:,9]
    
    age_tlabel = []
    for a in age_tlabel_:
        try:
            age_tlabel.append(int(a/10))
        except:
            age_tlabel.append(-1)
    age_tlabel = np.array(age_tlabel)

    icu_tlabel = []
    for a in icu_tlabel_:
        icu_tlabel.append(int(a<8))
    icu_tlabel = np.array(icu_tlabel)
    
    die_tlabel = []
    for a in die_tlabel_:
        die_tlabel.append(int(a<8))
    die_tlabel = np.array(die_tlabel)    
    
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
                         'query']]
    '''
    
    
    fig = plt.figure(figsize=(48,25),dpi=100)
    #plt.title('EDisease (Embedding of Disease)')    
    ax = fig.add_subplot(241)
    ax.scatter(T_tsne[:,0],T_tsne[:,1],
               marker = '.', alpha=alpha[0], label = 'EDisease')
    plt.legend()
    plt.title('2D t-SNE of EDisease (Embedding of Disease)') 
    
    ax2 = fig.add_subplot(242)
    for tril in range(2):
        cls = sex_tlabel==tril
        Gender ='F' if tril==0 else 'M'
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Gender= '+str(Gender)) 
    plt.legend()
    plt.title('EDisease (Gender)') 

    ax2 = fig.add_subplot(243)
    for tril in range(12):
        cls = age_tlabel==tril
        age = str(10*tril)+' ~ '+str(10*(tril+1)-1)
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Age= '+str(age)) 
    plt.legend()
    plt.title('EDisease (Age)')     

    ax2 = fig.add_subplot(244)
    for tril in range(3):
        cls = hosp_tlabel==tril
        if tril==0:
            hosp = 'Taipei'
        elif tril==1:
            hosp = 'Hsinchu'
        else:
            hosp = 'Yunlin'
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Hospital= '+str(hosp)) 
    plt.legend()
    plt.title('EDisease (Hospital)')     
    
    ax2 = fig.add_subplot(245)
    for tril in range(5):
        cls = triage_tlabel==tril+1
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'Triage= '+str(tril+1)) 
    plt.legend()
    plt.title('EDisease (Triage Level)') 
    
    ax3 = fig.add_subplot(246)
    for nohx in range(2):
        cls = hxnum_label==nohx
        wi = 'Without' if nohx==0 else 'With'
        ax3.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[0], label = 'EDisease_hx: '+str(nohx))    
    plt.legend()
    plt.title('EDisease (With/Without Medical History)')

    ax2 = fig.add_subplot(247)
    for tril in range(2):
        cls = icu_tlabel==tril
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[tril], label = 'ICU-7= '+str(tril)) 
    plt.legend()
    plt.title('EDisease (ICU in 7days)') 
    
    ax3 = fig.add_subplot(248)
    for nohx in range(2):
        cls = die_tlabel==nohx
        ax3.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[nohx], label = 'Death-7: '+str(nohx))    
    plt.legend()
    plt.title('EDisease (Death in 7days)')    
    
    #TODO sex, age, Hospital, icu, death
    
    plt.savefig('./pic/'+'EDisease_Oct.png')
    plt.show()
    plt.close()
    
    #======
    # 2nd figure
    #======
    
    fig = plt.figure(figsize=(22,10),dpi=100)
    #plt.title('EDisease (Embedding of Disease)')       
    ax2 = fig.add_subplot(121)
    for tril in range(2):
        cls = cva_tlabel==tril
        stroke ='N' if tril==0 else 'Y'
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[tril], label = 'Stroke= '+str(stroke)) 
    plt.legend()
    plt.title('EDisease (Stroke)') 

    ax2 = fig.add_subplot(122)
    for tril in range(2):
        cls = trm_tlabel==tril
        trauma ='N' if tril==0 else 'Y'
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=alpha[tril], label = 'Major Trauma= '+str(trauma)) 
    plt.legend()
    plt.title('EDisease (Major Trauma)')     

    plt.savefig('./pic/'+'EDisease_Disease.png')
    plt.show()
    plt.close()  
    
def plot_2d(result, alpha=[0.3,1]): 
    import matplotlib.pyplot as plt
    feature = result['target']

    idx = result['idx']
    all_label = result['label']['all_label']  
    
    triage_tlabel = all_label[:,1]    
    hosp_tlabel = all_label[:,2]
    icu_tlabel_ = all_label[:,3]
    die_tlabel_ = all_label[:,4]
    age_tlabel_ = all_label[:,5]
    sex_tlabel = all_label[:,6]
    cva_tlabel = all_label[:,7]
    trm_tlabel = all_label[:,8]
    query_tlabel = all_label[:,9]
    
    age_tlabel = []
    for a in age_tlabel_:
        try:
            age_tlabel.append(int(a/10))
        except:
            age_tlabel.append(-1)
    age_tlabel = np.array(age_tlabel)

    icu_tlabel = []
    for a in icu_tlabel_:
        icu_tlabel.append(int(a<8))
    icu_tlabel = np.array(icu_tlabel)
    
    die_tlabel = []
    for a in die_tlabel_:
        die_tlabel.append(int(a<8))
    die_tlabel = np.array(die_tlabel)    
    
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
                         'query']]
    '''
    
    dim_len = feature.shape[1]
    pic_len = int(dim_len/2)
    
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(pic_len):
        pn = '77{:.0f}'.format(p+1)
        xi = p%7
        yi = int(p/7)
        ax = plt.subplot2grid((7,7),(yi,xi))
        #ax = fig.add_subplot(int(pn))
        for tril in range(2):
            cls = sex_tlabel==tril        
            ax.scatter(feature[cls,2*p],feature[cls,2*p+1],
                       marker = '.', alpha=alpha[0])    
        plt.title('EDisease (Gender) '+str(p))     
                    
    plt.savefig('./pic/'+'EDisease_2d_sex.png')
    plt.show()
    plt.close()
    
    #======
    # 2nd figure
    #======
    
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(pic_len):
        pn = '77{:.0f}'.format(p+1)
        xi = p%7
        yi = int(p/7)
        ax = plt.subplot2grid((7,7),(yi,xi))
        for tril in range(2):
            cls = cva_tlabel==tril        
            ax.scatter(feature[cls,2*p],feature[cls,2*p+1],
                       marker = '.', alpha=alpha[tril])    
        plt.title('EDisease (Stroke) '+str(p))     
                   
    plt.savefig('./pic/'+'EDisease_2d_stroke.png')
    plt.show()
    plt.close()          
    
def plot_1(T_tsne,labels): 
    f_tlabel = labels['f_tlabel']
    triage_tlabel = labels['triage_tlabel']
    hxnum_label = labels['hxnum_label']
    
    fig = plt.figure(figsize=(36,10),dpi=100)
    #plt.title('EDisease (Embedding of Disease)')    
    ax = fig.add_subplot(131)
    ax.scatter(T_tsne[:,0],T_tsne[:,1],
               c=f_tlabel, marker = '.', alpha=0.5, label = 'EDisease')
    plt.legend()
    plt.title('2D t-SNE of EDisease (Embedding of Disease)')  
    ax2 = fig.add_subplot(132)
    for tril in range(5):
        cls = triage_tlabel==tril
        ax2.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=0.5, label = 'Triage= '+str(tril+1)) 
    plt.legend()
    plt.title('EDisease (Triage Level)') 
    ax3 = fig.add_subplot(133)
    for nohx in range(2):
        cls = hxnum_label==nohx
        ax3.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=0.5, label = 'EDisease_hx: '+str(nohx))    
    plt.legend()
    plt.title('EDisease (With/Without Medical History)')
    
    #TODO sex, age, Hospital, icu, death
    
    plt.savefig('./pic/'+'EDisease_tri.png')
    plt.show()
    plt.close()

def plot_2(T_tsne,S_tsne, f_tlabel, f_slabel):  
    fig = plt.figure(figsize=(22,10),dpi=100)
    ax = fig.add_subplot(121)
    ax.scatter(T_tsne[:,0],T_tsne[:,1],
               c=f_tlabel, marker = '.', alpha=0.5, label = 'Target')  
    ax2 = fig.add_subplot(122)
    ax2.scatter(T_tsne[:,0],T_tsne[:,1],
               marker = '.', alpha=0.4, label = 'Target: '+SourceTarget[DDT]['target'])  
    ax2.scatter(S_tsne[:,0],S_tsne[:,1],
               marker = '.', alpha=0.3, label = 'Source: '+SourceTarget[DDT]['source'])

    plt.title(DANN_name[DANN_type])
    plt.legend()
    plt.savefig('./DANN_Output/'+DANN_name[DANN_type]+'.jpg')
    plt.show()
    plt.close()

def draw_predict(sample, prob, gt, task='ICU',query=None):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    fontname = '../otf/NotoSansCJKtc-Regular.otf'
    myfont = font_manager.FontProperties(fname=fontname)
    #plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']
    plt.rcParams['axes.unicode_minus'] = False
    
    trg = sample['trg'][-5:-3]
    portal = sample['portal']
    
    txt = portal['CC']
    l = len(txt)
    
    hx = portal['HX']  
    ss = portal['SS']
    
    pic = np.ndarray((1150,1440,3),dtype='uint8')
    pic[:,:]=[255,255,255]
    cha_x = 20
    cha_y =50
    line_char_max = 45
    
    ss_char_max = 22

    mw = 110
    mh = 60
    mx = 80
    my = 50    
    
    def color_rank(prob): #prob = 100*heat  -100~100
        #[0,0,255] [0,127,255] [0,255,255] [0,255,127] [0,255,0] [255,255,0] [255,255,0] [255,127,0] [255,0,0]
        step = 3.5       
    
        if prob>=25:
            dis = abs(prob-25)
            cr = int(max(0,(255-dis*step)))
            return [255,cr,0]
        
        elif prob<25 and prob>=0:
            dis = abs(prob-0)
            gr = int(min(255,0+dis*step*3))
            return [gr,255,0]  
        
        elif prob<0 and prob>=-25:
            dis = abs(0-prob)
            gr = int(min(255,0+dis*step*3))
            return [0,255,gr]  
        
        elif prob<-25:
            dis = abs(-25-prob)
            cr = int(max(0,(255-dis*step)))
            return [0,cr,255] 
        
        else:
            return [128,128,128] 
    
    def prob2uint(iprob):
        return int(iprob*100)
    
    if prob is not None:
        for i in range(15):
            jj = i
            ii = ss_char_max
            pic[10+cha_y*(jj+4):30+cha_y*(jj+4),
                20+cha_x*0:20+cha_x*(ii+1)] = color_rank(5*i+5)   

            legend_y = 0 #20+50*(jj+1)
            legend_x = 740

        for i in range(-100,100):
            pic[legend_y+80:legend_y+90,legend_x+320+3*i:legend_x+320+3*(i+1)]=color_rank(i)

        for i in range(3):
            pic[my:my+mh,
                mx+(0+i)*mw:mx+(1+i)*mw] = color_rank(50*i+20)        
        
    fig = plt.figure(figsize=(10,20),dpi=(120))
    ax = fig.add_subplot(111)
    ax.imshow(pic)

    rect = plt.Rectangle((legend_x,legend_y+50),650,60,edgecolor='black',facecolor='none')
    ax.add_patch(rect)
    ax.text(1044,legend_y+70,'Heat',fontsize=7,style='italic')
    
    rect = plt.Rectangle((mx,my),mw,mh,edgecolor='black',facecolor='none')
    ax.add_patch(rect)    
    rect = plt.Rectangle((mx+1*mw,my),mw,mh,edgecolor='black',facecolor='none')
    ax.add_patch(rect)  
    rect = plt.Rectangle((mx+2*mw,my),mw,mh,edgecolor='black',facecolor='none')
    ax.add_patch(rect)  

       
    
    for i in range(11):
        if i < 1:
            ax.text(755+60*i,legend_y+105,'-1.0',fontsize=6)
        else:
            num = '{:.1f}'.format(float(-1.0+i/5))
            ax.text(755+60*i,legend_y+105,num,fontsize=6)
    
    ccc = 'Chief Complaint'
    for i in range(len(ccc)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(500+cha_x*(ii),
                120+cha_y*(jj+2),ccc[i],fontsize=12,color='black',
                horizontalalignment='center',
                verticalalignment='center')    

    for i in range(len(txt)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(550+cha_x*(ii),
                180+cha_y*(jj+2),txt[i],fontsize=8,color='black',
                horizontalalignment='center',
                verticalalignment='center',
                fontproperties=myfont)
    
    if task == 'ICU':
        mmm = f'{task}-7 Probability: {prob:.2f}'
        nnn = f'Transfer to {task} in {gt:.0f} days'
    elif task == 'Death':
        mmm = f'{task}-7 Probability: {prob:.2f}'
        nnn = f'{task} in {gt:.0f} days'
    elif task == 'Triage':
        mmm = f'Triage 1+2 Probability: {prob:.2f}'
        nnn = f'{task} was {gt:.0f}'
    for i in range(len(mmm)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(30+cha_x*(ii),
                20+cha_y*(jj),mmm[i],fontsize=12,color='black',
                horizontalalignment='center',
                verticalalignment='center')          
    for i in range(len(nnn)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(750+cha_x*(ii),
                20+cha_y*(jj),nnn[i],fontsize=12,color='black',
                horizontalalignment='center',
                verticalalignment='center')  
        
    mmm = 'MS   MCC  MHX'
    for i in range(len(mmm)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(132+cha_x*(ii),
                20+11+cha_y*(jj+1),mmm[i],fontsize=12,color='black',
                horizontalalignment='center',
                verticalalignment='center')          
        
        sskey = ['Age             ',
                 'Gender          ',
                 'Revisit Times   ',
                 'Systolic BP     ',
                 'Diastolic BP    ',
                 'Heart Rate      ',
                 'Saturation      ',
                 'Respiratory Rate',
                 'Body Temperature',
                 'Body Height     ',
                 'Body Weight     ',
                 'Pain Score      ',
                 'GCS(E)          ',
                 'GCS(V)          ',
                 'GCS(M)          ']
    ssfm = ['{:.0f}',['F','M'],'{:.0f}','{:.0f}','{:.0f}','{:.0f}',
            '{:.0f}','{:.0f}','{:.1f}','{:.0f}','{:.0f}','{:.0f}','{:.0f}','{:.0f}','{:.0f}']
    
    for s in range(15):
        if int(ss[s]) ==999:
            sstxt = sskey[s]+ ': N/A'
        else:
            if sskey[s] == 'Gender          ':
                sstxt = sskey[s]+ ': '+ssfm[s][int(ss[s])]
            else:
                sstxt = sskey[s]+ ': '+ssfm[s].format(ss[s])
                
        for i in range(len(sstxt)):
            jj = int(i/line_char_max)
            ii = i% line_char_max
            ax.text(30+cha_x*(ii),
                    120+cha_y*(jj+s+2),sstxt[i],fontsize=12,color='black',
                    horizontalalignment='center',
                    verticalalignment='center')  
            
    hhh = 'Past Medical History:'
    for i in range(len(hhh)):
        jj = int(i/line_char_max)
        ii = i% line_char_max
        ax.text(500+cha_x*(ii),
                260+cha_y*(jj+2),hhh[i],fontsize=12,color='black',
                horizontalalignment='center',
                verticalalignment='center')
    if hx is None or len(hx) == 0:
        sstxt = 'None'
        for i in range(len(sstxt)):
            jj = int(i/line_char_max)
            ii = i% line_char_max
            ax.text(550+cha_x*(ii),
                    320+cha_y*(jj+2),sstxt[i],fontsize=6,color='black',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontproperties=myfont)
    else:
        h_line = 0
        for h in range(len(hx)): 
            if len(hx) > 4:
                max_len_hx = 85
            else:
                max_len_hx = int(12/len(hx))*line_char_max-5
            sstxt = str(hx.iloc[h])
            if len(sstxt) >= max_len_hx: 
                sstxt = sstxt[:max_len_hx]+'...'
            for i in range(len(sstxt)):
                jj = int(i/line_char_max)
                ii = i% line_char_max
                ax.text(550+cha_x*(ii),
                        320+cha_y*(jj+h_line+2),sstxt[i],fontsize=6,color='black',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontproperties=myfont)
            h_line+=jj+1
            if h < len(hx) -1:
                sep = ' === '*7
                for i in range(len(sep)):
                    ii = i% line_char_max            
                    ax.text(600+cha_x*(ii),
                            320+cha_y*(h_line+2),sep[i],fontsize=6,color='green',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontproperties=myfont)
                h_line+=1
            if h > 3:
                sep = '... to history {:d}'.format(len(hx))
                for i in range(len(sep)):
                    ii = i% line_char_max            
                    ax.text(550+cha_x*(ii),
                            320+cha_y*(2+h_line),sep[i],fontsize=6,color='black',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontproperties=myfont)
                break       
        
    plt.axis('off')    
    if query is None:
        plt.savefig(f'./picSHOW/{task}_{100*gt:05.0f}_{str(portal["ACCOUNTIDSE2"])}.png')
    else:            
        path = f'./picSHOW/q_{query:05.0f}/'
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(path+f'{task}_{100*gt:05.0f}_{str(portal["ACCOUNTIDSE2"])}.png')    
    
def query(query_each,val_edisease):
    #use the MSE
    query_each = np.array(query_each)
    val_edisease = np.array(val_edisease)
    
    dist = np.mean((val_edisease-query_each)**2,axis=1)
    ars = np.argsort(dist)
    
    return ars    

def query_by_kmean(f_target,nclus=256):    
    from sklearn.cluster import KMeans    
    
    km = KMeans(n_clusters=nclus)
    y = km.fit(f_target)
    centro = y.cluster_centers_
    
    f_target_c = np.concatenate([centro,f_target],axis=0) 
    a = y.predict(f_target_c)

    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    T_tsne = tsne.fit_transform(f_target_c)    
    
    plot_kmean(T_tsne,a,10,nclus)
    
def plot_kmean(T_tsne,labels,n_clusters,draw_n): 
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10),dpi=100)
    #plt.title('EDisease (Embedding of Disease)')    
    ax = fig.add_subplot(111)
    
    centers, nt = T_tsne[:n_clusters],T_tsne[n_clusters:]
    centers_l, nl = labels[:n_clusters],labels[n_clusters:]
    for i in range(draw_n):
        fl = nl==i
        ax.scatter(nt[fl,0],nt[fl,1],
                   marker = '.', alpha=0.3, label = 'Cls'+str(i))
    plt.legend()
    
    for i in range(draw_n):
        fl = centers_l==i
        ax.scatter(centers[fl,0],centers[fl,1],
                   marker = '^', alpha=1, label = 'C'+str(i),linewidths=3)
    plt.legend()
    
    plt.title('2D t-SNE of EDisease (Kmean)')
    
    #TODO sex, age, Hospital, icu, death
    
    plt.savefig('./pic/'+'EDisease_kmean.png')
    plt.show()
    plt.close()    

    
    

    
    
def count_tsne_cls(net_F,baseBERT,cls_model, target, device,use_pi,ki,draw = False):
    from sklearn import manifold

    net_F.to(device)      
    net_F.eval()
    baseBERT.to(device)
    baseBERT.eval()
    cls_model.to(device)
    cls_model.eval()
    
    f_target_ = []
    triage_tlabel_ = []
    hxnum_label_ = []
    all_label_ = []
    idx_ =[]
    
    icu_ = []
    die_ = []
    tri_ = []
    poor_ = []
    poor2_ = []
    
    with torch.no_grad():
        t0 = time.time()
        for batch_idx, sample in enumerate(target):
            s_np = sample['structure'].numpy()
            c_np = sample['cc'].numpy()
            h_np = sample['ehx'].numpy()
            nans = np.all(np.isnan(s_np))
            nanc = np.all(np.isnan(c_np))
            nanh = np.all(np.isnan(h_np))
            
            trg = sample['trg']
            
            if ~nans and ~nanc and ~nanh:
                output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),_ , nohx, expand_data = net_F(baseBERT,sample,use_pi=use_pi,test=True)

                bs = len(s)
                # EDisease = output[:,:1]
                #using the hidden layer as EDisease, output for pretrain
                hidden = output
                f_target_.append(hidden[:,0].cpu())
                #print(output[:,0,:12].cpu())
                triage_tlabel_.append(s[:,-1].view(-1).cpu())
                hxnum_label_.append((nohx > 1).int().view(-1).cpu())
                all_label_.append(trg.cpu())
                idx_.append(sample['idx'])

                cls_icu,cls_die,cls_tri,cls_poor = cls_model(hidden[:,:1])
                
                cls_icu_soft = nn.Softmax(dim=-1)(cls_icu)
                cls_die_soft = nn.Softmax(dim=-1)(cls_die)
                cls_tri_soft = nn.Softmax(dim=-1)(cls_tri)
                cls_poor_soft = nn.Softmax(dim=-1)(cls_poor)
                               
                icu_.append(cls_icu_soft[:,-1].view(-1).cpu())
                die_.append(cls_die_soft[:,-1].view(-1).cpu())
                tri_.append(cls_tri_soft[:,-1].view(-1).cpu())
                poor_.append(cls_poor_soft[:,-1].view(-1).cpu())
                
                cls_poor2 = torch.max(cls_icu,cls_die)
                cls_poor_soft2 = nn.Softmax(dim=-1)(cls_poor2)
                poor2_.append(cls_poor_soft2[:,-1].view(-1).cpu())
                
                if batch_idx%100 ==0:
                    print('[{} ({:.0f}%)/ ep_time:{:.0f}min]'.format(batch_idx,
                            100. * batch_idx / len(target),
                            (time.time()-t0)*len(target)/(60*(batch_idx+1))))  
            else:
                print('nans: ', nans)
                print('nanc: ', nanc)
                print('nanh: ', nanh)
    
    f_target = torch.cat(f_target_,dim=0)
    
    icu = torch.cat(icu_,dim=0)
    die = torch.cat(die_,dim=0)
    tri = torch.cat(tri_,dim=0)
    poor = torch.cat(poor_,dim=0)
      
    triage_tlabel = torch.cat(triage_tlabel_,dim=0)
    hxnum_label = torch.cat(hxnum_label_,dim=0)
    f_tlabel = [0]*len(f_target)
    all_label = torch.cat(all_label_,dim=0)
    idx = torch.cat(idx_,dim=0)
    print(f_target.shape)
    
    labels = {'f_tlabel':f_tlabel,
              'triage_tlabel':triage_tlabel.view(-1).cpu().numpy(),
              'hxnum_label':hxnum_label.view(-1).cpu().numpy(),
              'all_label':all_label.numpy()}
    
    pred = {'icu':icu,
            'die':die,
            'tri':tri,
            'poor':poor
           }
    
    print(' ** draw ROC AUC ** ')
    # plot_roc(pred,labels)
    roc_auc2 = count_roc_nhamcs2(pred,labels,ep=ki,draw=draw)
    
    result ={'target':f_target.numpy(),
             'idx':idx.cpu().numpy(),
             'label':labels,
             'pred':pred}
    return result, roc_auc2
    

def count_roc_nhamcs2(pred,labels,ep=0,draw = False):
    from sklearn.metrics import roc_curve, auc
    ROC_threshold = torch.linspace(0,1,100).numpy()
    
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
    
    pred_icu = pred['icu']
    pred_die = pred['die']
    pred_poor = pred['poor']
    
    rocset = ['Critical']
    roc_label = [poor_tlabel]
    roc_pred = [pred_poor]
    
    fig = plt.figure(figsize=(10,10),dpi=200)
    
    for i, s in enumerate(rocset):
        pn = '11{:.0f}'.format(i+1)
  
        ax = fig.add_subplot(pn)
        fpr, tpr, _ = roc_curve(roc_label[i], roc_pred[i].numpy())
        roc_auc2 = auc(fpr,tpr)

        label_auc2 = str(s)+', AUROC='+str(roc_auc2)
        ax.plot(fpr,tpr,label=label_auc2)
        ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')

        ax.set_xlabel('1-specificity')
        ax.set_ylabel('sensitivity')
        ax.set_title('ROC curve without pretrain')
        plt.legend()
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.legend()
    if draw:
        plt.savefig('./pic_ROC/AUC_nhamcs_'+str(ep)+'.png')
    return roc_auc2
  
    

    
    

    
    
