import pandas as pd
import numpy as np
import glob
import os
import random
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
import torch

# === impoert BERT ===
BERT_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#BERT_model = BertModel.from_pretrained("bert-base-multilingual-cased")
# ====================

def load_datas(preprocessing=True):
    print('load_datas')
    datas = pd.read_csv('NHAMCS_2007_17_RA.csv', sep='\t',dtype=object)
    
    datas['HEIGHT'] = None
    datas['WEIGHT'] = None
    datas['BE'] = None
    datas['BV'] = None
    datas['BM'] = None
    # datas['GCS'] = None
    
    datas = datas.reset_index()
    
    temp = datas[datas['AGE']=='Under one year']
    datas.at[temp.index, 'AGE'] = '0.0'
    temp = datas[datas['AGE']=='93 years and over']
    datas.at[temp.index, 'AGE'] = '93.0'
    
    datas['SEX'] = datas['SEX'].str.replace('Female','0')
    datas['SEX'] = datas['SEX'].str.replace('Male','1')
    # datas['sex_code'] = datas['SEX'].cat.codes
    
    temp = datas[datas['BPSYS']=='Blank']
    datas.at[temp.index, 'BPSYS'] = None
    
    temp = datas[datas['BPDIAS']=='Blank']
    datas.at[temp.index, 'BPDIAS'] = None
    
    temp = datas[datas['PULSE']=='Blank']
    datas.at[temp.index, 'PULSE'] = None
    
    temp = datas[datas['POPCT']=='Blank']
    datas.at[temp.index, 'POPCT'] = None
    
    temp = datas[datas['RESPR']=='Blank']
    datas.at[temp.index, 'RESPR'] = None
    
    temp = datas[datas['TEMPF']=='Blank']
    datas.at[temp.index, 'TEMPF'] = None
    
    temp = datas[datas['PAINSCALE']=='Unknown']
    datas.at[temp.index, 'PAINSCALE'] = None
    temp = datas[datas['PAINSCALE']=='Blank']
    datas.at[temp.index, 'PAINSCALE'] = None
    
    temp = datas[datas['GCS']=='Blank']
    datas.at[temp.index, 'GCS'] = None
    
    
    datas['AGE'] = pd.to_numeric(datas['AGE'],errors='coerce')
    datas['SEX'] = pd.to_numeric(datas['SEX'],errors='coerce')
    # datas['ACCOUNTSEQNO'] = pd.to_numeric(datas['ACCOUNTSEQNO'],errors='coerce')
    datas['BPSYS'] = pd.to_numeric(datas['BPSYS'],errors='coerce')
    datas['BPDIAS'] = pd.to_numeric(datas['BPDIAS'],errors='coerce')
    datas['PULSE'] = pd.to_numeric(datas['PULSE'],errors='coerce')
    
    datas['POPCT'] = pd.to_numeric(datas['POPCT'],errors='coerce')
    datas['RESPR'] = pd.to_numeric(datas['RESPR'],errors='coerce')
    datas['TEMPF'] = pd.to_numeric(datas['TEMPF'],errors='coerce')
    datas['PAINSCALE'] = pd.to_numeric(datas['PAINSCALE'],errors='coerce')
    
    datas['HEIGHT'] = pd.to_numeric(datas['HEIGHT'],errors='coerce')
    datas['WEIGHT'] = pd.to_numeric(datas['WEIGHT'],errors='coerce')
    datas['BE'] = pd.to_numeric(datas['BE'],errors='coerce')
    datas['BV'] = pd.to_numeric(datas['BV'],errors='coerce')
    datas['BM'] = pd.to_numeric(datas['BM'],errors='coerce')
    datas['GCS'] = pd.to_numeric(datas['GCS'],errors='coerce')
    
    datas['TEMPF'] = (datas['TEMPF'] - 32)/1.8

    datas['DIEDED'] = datas['DIEDED'].str.replace('No','9999')
    datas['DIEDED'] = datas['DIEDED'].str.replace('Yes','1')
    
    datas['DOA'] = datas['DOA'].str.replace('No','0')
    datas['DOA'] = datas['DOA'].str.replace('Yes','1')

    datas['DIEDED'] = pd.to_numeric(datas['DIEDED'],errors='coerce')
    datas['DOA'] = pd.to_numeric(datas['DOA'],errors='coerce')
    
    datas['ICU'] = 9999
    
    temp = datas[datas['ADMIT']=='Critical care unit']
    datas.at[temp.index,'ICU'] = 1
    
    datas['AMA'] = pd.to_numeric(datas['AMA'],errors='coerce')

                    
    S_select = ['AGE',
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
    
    
    H_select = ['EDDIAL', 'DIABETES', 'DEMENTIA', 'MIHX', 'DVT',
            'CANCER', 'ETOHAB', 'ALZHD', 'ASTHMA', 'CEBVD', 'CKD', 'COPD',
            'CHF', 'CAD', 'DEPRN', 'DIABTYP1', 'DIABTYP2', 'DIABTYP0', 'ESRD',
            'HPE', 'EDHIV', 'HYPLIPID', 'HTN', 'OBESITY', 'OSA', 'OSTPRSIS','SUBSTAB',]
    
    target_select = ['DOA','DIEDED','ADMIT']
    
    datas['padding'] = ' '
    datas['CHIEFCOMPLAIN'] = datas['RFV1'].fillna('')+datas['padding']+datas['RFV2'].fillna('')+datas['padding']+datas['RFV3'].fillna('')+datas['padding']+datas['RFV4'].fillna('')+datas['padding']+datas['RFV5'].fillna('')
    datas['CHIEFCOMPLAIN'] = datas['CHIEFCOMPLAIN'].str.replace('Blank','')
    
    ALL_SELECT = S_select+['CHIEFCOMPLAIN',]+H_select+target_select
    
    # ==================================

    # ==================================
    # data clean-up
    def data_preprocessing_clean_up(df):
        print('origin case num = ', len(df))
        # clean 0 ChiefComplaints = NaN
        df_temp = df[df.CHIEFCOMPLAIN.isna()]
        df = df.drop(df_temp.index)
        print('clean ChiefComplaints = NaN, case num = ', len(df))        
        
        select = ['AGE',
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
        
        df_clean = df[select]
        df_clean = df_clean.abs()

        print('origin case num = ', len(df_clean))
        # clean A BPDIAS > BPSYS
        df_clean_temp = df_clean[df_clean.BPDIAS > df_clean.BPSYS]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.BPDIAS == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)        
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean DBP > SBP, case num = ', len(df_clean))

        # clean B BPSYS > 300
        df_clean_temp = df_clean[df_clean.BPSYS>300]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.BPSYS == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)  
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean SBP > 300, case num = ', len(df_clean))   

        # clean C BPDIAS > 300
        df_clean_temp = df_clean[df_clean.BPDIAS>300]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.BPDIAS == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)  
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean DBP > 300, case num = ', len(df_clean))

        # clean D PULSE > 250
        df_clean_temp = df_clean[df_clean.PULSE > 250]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.PULSE == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)  
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean PULSE > 250, case num = ', len(df_clean))

        # clean E RESPR > 50
        df_clean_temp = df_clean[df_clean.RESPR > 50]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.RESPR == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)  
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean RESPR > 50, case num = ', len(df_clean))

        # clean F TEMPF > 48
        df_clean_temp = df_clean[df_clean.TEMPF > 48]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.TEMPF == 999]
 #       df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index) 
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean TEMPF > 48, case num = ', len(df_clean))

        # clean G TEMPF < 10
        df_clean_temp = df_clean[df_clean.TEMPF < 10]
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean TEMPF < 10, case num = ', len(df_clean))

        # clean H HEIGHT > 250 | (HEIGHT < 50 + AGE >1) | (HEIGHT <20)
        df_clean_temp = df_clean[(df_clean.HEIGHT > 250) | ((df_clean['HEIGHT']<50)&(df_clean['AGE']>1)) | (df_clean['HEIGHT']<20)]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.HEIGHT == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean HEIGHT > 250 | (HEIGHT < 50 + AGE >1) | (HEIGHT <20), case num = ', len(df_clean))

        # clean H WEIGHT > 400 | (WEIGHT < 1 + AGE > 0) | (WEIGHT < 3 + AGE > 1)
        df_clean_temp = df_clean[(df_clean.WEIGHT > 400) | ((df_clean['WEIGHT']<1)&(df_clean['AGE']>0)) | ((df_clean['WEIGHT']<3)&(df_clean['AGE']>1))]
#        df_clean_temp_999 = df_clean_temp[df_clean_temp.WEIGHT == 999]
#        df_clean_temp = df_clean_temp.drop(df_clean_temp_999.index)
        df_clean = df_clean.drop(df_clean_temp.index)
        df = df.drop(df_clean_temp.index)
        print('clean WEIGHT > 400 | (WEIGHT < 1 + AGE > 0) | (WEIGHT < 3 + AGE > 1), case num = ', len(df_clean))

        df[select] = df_clean

        return df


    # ==================================
    datas = data_preprocessing_clean_up(datas)
    datas['query'] = 0
    datas['idx'] = np.arange(len(datas))
  
    datas.at[datas[datas['AGE']==999].index,['AGE']]=None 
    

    print('datas, case num = ', len(datas))

    # df_clean_temp = datas[datas.AGE<18]
    # datas = datas.drop(df_clean_temp.index)
    # print('clean AGE< 18, case num = ', len(datas))

    df_clean_temp = datas[datas.DOA>0]
    datas = datas.drop(df_clean_temp.index)
    print('clean OHCA, case num = ', len(datas))
    
    df_clean_temp = datas[datas.LEFTAMA=='Yes']
    datas = datas.drop(df_clean_temp.index)
    print('clean LEFTAMA, case num = ', len(datas))
    
    df_clean_temp = datas[datas['DIAG1R']=='209910.0']
    datas = datas.drop(df_clean_temp.index)
    print('clean Left before Seen 07-15, case num = ', len(datas))
    df_clean_temp = datas[datas['AMA']>0]
    datas = datas.drop(df_clean_temp.index)
    print('clean Left before Seen 16-17, case num = ', len(datas))
    
    
    
    print('datas, case num = ', len(datas))
    
    datas_train = datas.sample(frac=0.8, random_state=11)
    datas_val = datas.drop(datas_train.index)
    datas_test = datas_val.sample(frac=0.5, random_state=11)
    datas_val = datas_val.drop(datas_test.index)
    
    print('  ***** ================ *****  ')
    print('datas_train, case num = ', len(datas_train))
    print('datas_val, case num = ', len(datas_val))
    print('datas_test, case num = ', len(datas_test))

    print('all SEX: ', datas['SEX'].value_counts())
    print('trainset SEX: ', datas_train['SEX'].value_counts())
    print('valset SEX: ', datas_val['SEX'].value_counts())
    print('testset SEX: ', datas_test['SEX'].value_counts())

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
    dtemp = datas[structural]
    dtemp_mean = dtemp.mean(axis=0, skipna=True)
    dtemp_std = dtemp.std(axis=0, skipna=True)
    dtemp_normalization = pd.concat([dtemp_mean,dtemp_std],axis=1)
    print('all',dtemp_normalization)
        
    dm = datas_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T
    
    dtemp = datas_val[structural]
    dtemp_mean = dtemp.mean(axis=0, skipna=True)
    dtemp_std = dtemp.std(axis=0, skipna=True)
    dtemp_normalization = pd.concat([dtemp_mean,dtemp_std],axis=1)
    print('valset',dtemp_normalization)

    dtemp = datas_test[structural]
    dtemp_mean = dtemp.mean(axis=0, skipna=True)
    dtemp_std = dtemp.std(axis=0, skipna=True)
    dtemp_normalization = pd.concat([dtemp_mean,dtemp_std],axis=1)
    print('testset',dtemp_normalization)    
    # ==================================
    
    datas = {'datas_train':datas_train,
             'datas_val':datas_val,
             'datas_test':datas_test,
             'dm_normalization_np':dm_normalization_np,
             'datas':datas,
            }
    
    return datas


'''
origin case num =  305897
clean ChiefComplaints = NaN, case num =  305897
origin case num =  305897
clean DBP > SBP, case num =  305896
clean SBP > 300, case num =  305896
clean DBP > 300, case num =  305896
clean PULSE > 250, case num =  305896
clean RESPR > 50, case num =  304257
clean TEMPF > 48, case num =  304257
clean TEMPF < 10, case num =  304257
clean HEIGHT > 250 | (HEIGHT < 50 + AGE >1) | (HEIGHT <20), case num =  304257
clean WEIGHT > 400 | (WEIGHT < 1 + AGE > 0) | (WEIGHT < 3 + AGE > 1), case num =  304257
datas, case num =  304257
clean OHCA, case num =  304150
clean LEFTAMA, case num =  301014
clean Left before Seen 07-15, case num =  297819
clean Left before Seen 16-17, case num =  297508
datas, case num =  297508
  ***** ================ *****  
datas_train, case num =  238006
datas_val, case num =  29751
datas_test, case num =  29751
all SEX:  0.0    162361
1.0    135147
Name: SEX, dtype: int64
trainset SEX:  0.0    129846
1.0    108160
Name: SEX, dtype: int64
valset SEX:  0.0    16231
1.0    13520
Name: SEX, dtype: int64
testset SEX:  0.0    16284
1.0    13467
Name: SEX, dtype: int64
all                     0          1
AGE         37.433454  23.990922
SEX          0.454263   0.497905
GCS         14.569873   2.054528
BPSYS      132.949637  23.523146
BPDIAS      77.716171  14.579535
PULSE       91.083979  22.810041
POPCT       97.297692   6.199818
RESPR       19.314817   4.351809
TEMPF       36.808739   0.633604
HEIGHT            NaN        NaN
WEIGHT            NaN        NaN
PAINSCALE    4.805838   3.690409
BE                NaN        NaN
BV                NaN        NaN
BM                NaN        NaN
trainset                     0          1
AGE         37.404568  23.996438
SEX          0.454442   0.497921
GCS         14.576636   2.042085
BPSYS      132.954078  23.536278
BPDIAS      77.730627  14.591664
PULSE       91.122740  22.834695
POPCT       97.295778   6.199257
RESPR       19.322907   4.358769
TEMPF       36.808785   0.633791
HEIGHT            NaN        NaN
WEIGHT            NaN        NaN
PAINSCALE    4.799518   3.691087
BE                NaN        NaN
BV                NaN        NaN
BM                NaN        NaN
valset                     0          1
AGE         37.558548  24.040220
SEX          0.454439   0.497928
GCS         14.553728   2.081834
BPSYS      132.831644  23.464692
BPDIAS      77.587410  14.518639
PULSE       91.086966  22.678532
POPCT       97.289328   6.217505
RESPR       19.293165   4.292115
TEMPF       36.808011   0.630878
HEIGHT            NaN        NaN
WEIGHT            NaN        NaN
PAINSCALE    4.850716   3.706285
BE                NaN        NaN
BV                NaN        NaN
BM                NaN        NaN
testset                     0          1
AGE         37.539437  23.897367
SEX          0.452657   0.497762
GCS         14.531550   2.125892
BPSYS      133.031807  23.477061
BPDIAS      77.729348  14.543302
PULSE       90.770938  22.741867
POPCT       97.321337   6.186762
RESPR       19.271704   4.355162
TEMPF       36.809099   0.634847
HEIGHT            NaN        NaN
WEIGHT            NaN        NaN
PAINSCALE    4.811372   3.668938
BE                NaN        NaN
BV                NaN        NaN
BM                NaN        NaN
'''

class EDEW_Dataset(Dataset):
    def __init__(self, 
                 ds,
                 tokanizer,
                 normalization,
                 dsidx=None):
        self.ds = ds
        self.tokanizer = tokanizer
        self.normalization = normalization
        self.dsidx = dsidx
        
        if dsidx is None:
            self.len = len(ds)
        else:
            self.len = len(dsidx)
    
    def __getitem__(self, index):
        if self.dsidx is None:
            e_patient = self.ds.iloc[index]
        else: 
            e_patient = self.ds.iloc[self.dsidx.iloc[index]]        
       
        chief_complaint = e_patient['CHIEFCOMPLAIN']
        cc_tokens = self.tokanizer.tokenize(str(chief_complaint))
        # add BERT cls head
        cc_tokens = [self.tokanizer.cls_token, *cc_tokens]
        cc_tokens = cc_tokens[:512]
        cc_token_ids = self.tokanizer.convert_tokens_to_ids(cc_tokens)

        H_select = ['EDDIAL', 'DIABETES', 'DEMENTIA', 'MIHX', 'DVT',
            'CANCER', 'ETOHAB', 'ALZHD', 'ASTHMA', 'CEBVD', 'CKD', 'COPD',
            'CHF', 'CAD', 'DEPRN', 'DIABTYP1', 'DIABTYP2', 'DIABTYP0', 'ESRD',
            'HPE', 'EDHIV', 'HYPLIPID', 'HTN', 'OBESITY', 'OSA', 'OSTPRSIS','SUBSTAB',]

        H_content = ['Dialysis, HD, PD, H/D, P/D, ',
                     'Diabetes DM, ',
                     'Dementia, ',
                     'myocardial infarction (MI) (CAD), ',
                     'deep vein thrombosis (DVT), ',
                     'Cancer, ', 
                     'Alcoholism, Alcohol misuse, abuse, dependence, ', 
                     'Alzheimer’s disease, dementia, ',
                     'Asthma, ', 
                     'Cerebrovascular disease, stroke (CVA), transient ischemic attack (TIA), ', 
                     'Chronic kidney disease (CKD), ', 
                     'Chronic obstructive pulmonary disease (COPD), ',
                     'Congestive heart failure, CHF, ', 
                     'Coronary artery disease (CAD), ischemic heart disease (IHD), myocardial infarction (MI), ', 
                     'Depression, ', 
                     'Diabetes mellitus (DM) Type I',
                     'Diabetes mellitus (DM) Type 2', 
                     'Diabetes mellitus (DM)', 
                     'End-stage renal disease (ESRD), ',
                     'Pulmonary embolism (PE), deep vein thrombosis (DVT), venous thromboembolism (VTE), ', 
                     'HIV infection, AIDS',
                     'Hyperlipidemia, ', 
                     'Hypertension, ', 
                     'Obesity, ',
                     'Obstructive sleep apnea (OSA), ', 
                     'Osteoporosis, ',
                     'Substance abuse or dependence, ',]
        
        hxyn = e_patient[H_select].values
        
        history = ''
        
        for j, y in enumerate(hxyn):
            if y =='Yes' or y =='YES' or y =='yes':
                history += H_content[j]
                
        hx_token_ids = []
        hx_token_ids.append(torch.tensor([101,0,0,0,0],dtype=torch.float32)) 
               
        
        hx_tokens = self.tokanizer.tokenize(history)
        # add BERT cls head
        while len(hx_tokens)>511:
            hx_tokens_i,hx_tokens = hx_tokens[:511],hx_tokens[511:]
            hx_tokens_i = [self.tokanizer.cls_token, *hx_tokens_i]
            hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens_i)
            hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        hx_tokens = [self.tokanizer.cls_token, *hx_tokens]
        hx_tokens = hx_tokens[:512]
        hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens)
        hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        

        # ee_patient = e_patient.reshape(1,-1) 
        #          0   1      2      3   4  5  6   7 8  9 10  11     12      13      14
    # structure = y/o, sex, revisit,SBP,DBP,P,SPO2,R,T, H, W,Pain,Triage_C, Triage_H
    # structure = y/o, sex, revisit,SBP,DBP,P,SPO2,R,T, H, W,Pain,   BE,     BV      BM
        S_select = ['AGE',
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
        structure_df = e_patient[S_select]
        
        structure = np.array(structure_df,dtype='float')
        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = 999
                    
        nan = np.isnan(structure)
                    
        # # normalization
        structure_np = np.float32(structure)
        structure_mean = self.normalization[0]
#        structure_mean[0] = 43.34
        structure_mean[1] = 0.5
        structure_mean[2] = 0
        structure_std  = self.normalization[1]
#        structure_std[0] = 26.97
        structure_std[1] = 0.5
        structure_std[2] = 1
        structure_normalization = (structure_np-structure_mean)/(structure_std+1e-6)
        
        structure_normalization[nan] = 0
        nan = structure_np == 999       
        structure_normalization[nan] = 0
        if structure_np[3] == 999:
            structure_normalization[3] = -4
        if structure_np[4] == 999:
            structure_normalization[4] = -4
            
        structure_tensor = torch.tensor(np.float32(structure_normalization),dtype=torch.float32)
        chief_complaint_tensor = torch.tensor(cc_token_ids,dtype=torch.float32)

        target_select = ['DOA',
                         'DIEDED',
                         'DOA',
                         'ICU',
                         'DIEDED',
                         'AGE',
                         'SEX',
                         'DOA',
                         'DOA',
                         'query']
        
        trg = e_patient[target_select]
               
        trg_tensor = torch.tensor(trg,dtype=torch.float32)

        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = -1
        
        portal = {'CC':str(chief_complaint),
                  'HX':history,
                  'SS':structure}
        
        datas = {'structure': structure_tensor,
                 'cc': chief_complaint_tensor,
                 'hx': hx_token_ids,
                 'trg':trg_tensor,
                 'portal':portal,
                 'idx':e_patient['idx']
                 }             
          
        return datas
    
    def __len__(self):
        return self.len

def collate_fn(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    structure = [DD['structure'] for DD in datas]
    cc = [DD['cc'] for DD in datas]
    stack_hx_ = [DD['hx'] for DD in datas]
    trg = [DD['trg'] for DD in datas]
    idx = [DD['idx'] for DD in datas]
    
    stack_hx = []
    hx_n = []
    for shx in stack_hx_:
        hx_n.append(len(shx))
        for eshx in shx:
            stack_hx.append(eshx)
       
    ehx = stack_hx
    origin_cc_length = [len(d) for d in cc]
    origin_ehx_length = [len(d) for d in ehx]
    # origin_pi_length = [len(d) for d in pi]
    
    cc = pad_sequence(cc,
                      batch_first = True,
                      padding_value=0)
    ehx = pad_sequence(ehx,
                      batch_first = True,
                      padding_value=0)

    
    mask_padding_cc = torch.zeros(cc.shape,dtype=torch.long)
    mask_padding_ehx = torch.zeros(ehx.shape,dtype=torch.long)
    
    for i,e in enumerate(origin_cc_length):
        mask_padding_cc[i,:e] = 1
    for i,e in enumerate(origin_ehx_length):
        mask_padding_ehx[i,:e] = 1
        
    batch['structure'] = torch.stack(structure)
    batch['cc'] = cc
    batch['ehx'] = ehx
    
    batch['mask_cc'] = mask_padding_cc
    batch['mask_ehx'] = mask_padding_ehx
    
    batch['stack_hx_n'] = torch.tensor(hx_n)
    batch['origin_cc_length'] = torch.tensor(origin_cc_length)
    batch['origin_ehx_length'] = torch.tensor(origin_ehx_length)
    
    batch['trg'] = torch.stack(trg)
    
    batch['idx'] = torch.tensor(idx)
    
    return batch


class pickle_Dataset(Dataset):
    def __init__(self, 
                 ds,
                 tokanizer,
                 normalization,
                 dsidx=None):
        self.ds = ds
        self.tokanizer = tokanizer
        self.normalization = normalization
        self.dsidx = dsidx
        
        if dsidx is None:
            self.len = len(ds)
        else:
            self.len = len(dsidx)
    
    def __getitem__(self, index):
        if self.dsidx is None:
            e_patient = self.ds.iloc[index]
        else: 
            e_patient = self.ds.iloc[self.dsidx.iloc[index]]        
       
        chief_complaint = e_patient['CHIEFCOMPLAIN']
        
        ccemb = e_patient['ccemb']
        hxemb = e_patient['hxemb']

        S_select = ['AGE',
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
        structure_df = e_patient[S_select]
        
        structure = np.array(structure_df,dtype='float')
        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = 999
                    
        nan = np.isnan(structure)
                    
        # # normalization
        structure_np = np.float32(structure)
        structure_mean = self.normalization[0]
#        structure_mean[0] = 43.34
        structure_mean[1] = 0.5
        structure_mean[2] = 0
        structure_std  = self.normalization[1]
#        structure_std[0] = 26.97
        structure_std[1] = 0.5
        structure_std[2] = 1
        structure_normalization = (structure_np-structure_mean)/(structure_std+1e-6)
        
        structure_normalization[nan] = 0
        nan = structure_np == 999       
        structure_normalization[nan] = 0
        if structure_np[3] == 999:
            structure_normalization[3] = -4
        if structure_np[4] == 999:
            structure_normalization[4] = -4
            
        structure_tensor = torch.tensor(np.float32(structure_normalization),dtype=torch.float32)
        # chief_complaint_tensor = torch.tensor(cc_token_ids,dtype=torch.float32)

        target_select = ['DOA',
                         'DIEDED',
                         'DOA',
                         'ICU',
                         'DIEDED',
                         'AGE',
                         'SEX',
                         'DOA',
                         'DOA',
                         'query']
        
        trg = e_patient[target_select]
               
        trg_tensor = torch.tensor(trg,dtype=torch.float32)
        piemb = torch.tensor([0])

        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = -1
        
        # portal = {'CC':str(chief_complaint),
        #           'HX':history,
        #           'SS':structure}
        
        datas = {'structure': structure_tensor,
                 'ccemb': ccemb,
                 'hxemb': hxemb,
                 'piemb': piemb,
                 # 'cc': chief_complaint_tensor,
                 # 'hx': hx_token_ids,
                 'trg':trg_tensor,
                 'stack_hx_n':torch.tensor([5]),
                 # 'portal':portal,
                 # 'idx':e_patient['idx']
                 }             
          
        return datas
    
    def __len__(self):
        return self.len



if __name__ == '__main__':

    all_datas = load_datas()
    datas_train = all_datas['datas_train']
    # data01_person = all_datas['data01_person']
    # data02_wh = all_datas['data02_wh']
    # data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']
    # datas_train = all_datas['datas_train']
    
    datas_val = all_datas['datas_val']
    datas_test = all_datas['datas_test']
    
    dat = all_datas['datas']
    
    EDEW_DS = EDEW_Dataset(ds= datas_train,          
                           tokanizer= BERT_tokenizer,
                           normalization = dm_normalization_np,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=0,
                         batch_size=4,
                         collate_fn=collate_fn)
'''
    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=8,
                         batch_size=8,
                         collate_fn=collate_fn)
    
    
    for batch_idx, sample in enumerate(EDEW_DL):    
        structure = sample['structure']
        cc = sample['cc'] 
        ehx = sample['ehx']

        mask_padding_cc = sample['mask_cc'] 
        mask_padding_ehx = sample['mask_ehx'] 

        hx_n = sample['stack_hx_n']
        origin_cc_length = sample['origin_cc_length']
        origin_dhx_length = sample['origin_ehx_length']

        print(batch_idx)

        if batch_idx > 5:
            break
        
    
    print(structure)
    print(cc)
    print(ehx)
    print(mask_padding_cc)
    print(mask_padding_ehx)
    print(hx_n)
    print(origin_cc_length)
    print(origin_dhx_length)
    
'''    
    
    