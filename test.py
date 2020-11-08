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


print('load_datas')
filepath = './NHAMCS_DATA'
data_list = glob.glob(os.path.join(filepath, '*'))
data_list.sort()

data = []

for file in data_list:
    print(file)
    data.append(pd.read_spss(file,convert_categoricals=True))
datas = pd.concat(data,axis= 0)

AA2017 = data[-1]
AA2016 = data[-2]

AA2017['AMA'] = 0
AA2016['AMA'] = 0

temp = AA2017[AA2017['DIAG1'].cat.codes == 1039]
AA2017.at[temp.index, 'AMA'] = 1

temp = AA2017[AA2017['HDDIAG1'].cat.codes == 382]
AA2017.at[temp.index, 'AMA'] = 1

temp = AA2016[AA2016['DIAG1'].cat.codes == 921]
AA2016.at[temp.index, 'AMA'] = 1

temp = AA2016[AA2016['HDDIAG1'].cat.codes == 295]
AA2016.at[temp.index, 'AMA'] = 1

data[-1] = AA2017
data[-2] = AA2016

datas.to_csv('NHAMCS_2007_17_RA.csv', sep='\t')

datas = pd.read_csv('NHAMCS_2007_17_RA.csv', sep='\t',dtype=object)

# datas['HEIGHT'] = None
# datas['WEIGHT'] = None
# datas['BE'] = None
# datas['BV'] = None
# datas['BM'] = None
# # datas['GCS'] = None

# datas = datas.reset_index()

# temp = datas[datas['AGE']=='Under one year']
# datas.at[temp.index, 'AGE'] = '0.0'
# temp = datas[datas['AGE']=='93 years and over']
# datas.at[temp.index, 'AGE'] = '93.0'

# datas['SEX'] = datas['SEX'].str.replace('Female','0')
# datas['SEX'] = datas['SEX'].str.replace('Male','1')

# temp = datas[datas['BPSYS']=='Blank']
# datas.at[temp.index, 'BPSYS'] = None

# temp = datas[datas['BPDIAS']=='Blank']
# datas.at[temp.index, 'BPDIAS'] = None

# temp = datas[datas['PULSE']=='Blank']
# datas.at[temp.index, 'PULSE'] = None

# temp = datas[datas['POPCT']=='Blank']
# datas.at[temp.index, 'POPCT'] = None

# temp = datas[datas['RESPR']=='Blank']
# datas.at[temp.index, 'RESPR'] = None

# temp = datas[datas['TEMPF']=='Blank']
# datas.at[temp.index, 'TEMPF'] = None

# temp = datas[datas['PAINSCALE']=='Unknown']
# datas.at[temp.index, 'PAINSCALE'] = None
# temp = datas[datas['PAINSCALE']=='Blank']
# datas.at[temp.index, 'PAINSCALE'] = None

# temp = datas[datas['GCS']=='Blank']
# datas.at[temp.index, 'GCS'] = None


# datas['AGE'] = pd.to_numeric(datas['AGE'],errors='coerce')
# datas['SEX'] = pd.to_numeric(datas['SEX'],errors='coerce')
# # datas['ACCOUNTSEQNO'] = pd.to_numeric(datas['ACCOUNTSEQNO'],errors='coerce')
# datas['BPSYS'] = pd.to_numeric(datas['BPSYS'],errors='coerce')
# datas['BPDIAS'] = pd.to_numeric(datas['BPDIAS'],errors='coerce')
# datas['PULSE'] = pd.to_numeric(datas['PULSE'],errors='coerce')

# datas['POPCT'] = pd.to_numeric(datas['POPCT'],errors='coerce')
# datas['RESPR'] = pd.to_numeric(datas['RESPR'],errors='coerce')
# datas['TEMPF'] = pd.to_numeric(datas['TEMPF'],errors='coerce')
# datas['PAINSCALE'] = pd.to_numeric(datas['PAINSCALE'],errors='coerce')

# datas['HEIGHT'] = pd.to_numeric(datas['HEIGHT'],errors='coerce')
# datas['WEIGHT'] = pd.to_numeric(datas['WEIGHT'],errors='coerce')
# datas['BE'] = pd.to_numeric(datas['BE'],errors='coerce')
# datas['BV'] = pd.to_numeric(datas['BV'],errors='coerce')
# datas['BM'] = pd.to_numeric(datas['BM'],errors='coerce')
# datas['GCS'] = pd.to_numeric(datas['GCS'],errors='coerce')

# # datas['AGE'].str.replace('Under one year', '0.0', regex=True)
            
# S_select = ['AGE',
#             'SEX',
#             'GCS',
#             'BPSYS',
#             'BPDIAS',
#             'PULSE',
#             'POPCT',
#             'RESPR',
#             'TEMPF',
#             'HEIGHT', 
#             'WEIGHT',
#             'PAINSCALE',
#             'BE',
#             'BV',
#             'BM'
#             ]

# '''
# 'CEBVD', 'CHF', 'EDDIAL', 'EDHIV', 'DIABETES',

# 'CANCER', 'COPD', 'DEMENTIA', 'MIHX', 'DVT',


# myocardial infarction

# '''
# H_select = ['EDDIAL', 'DIABETES', 'DEMENTIA', 'MIHX', 'DVT',
#             'CANCER', 'ETOHAB', 'ALZHD', 'ASTHMA', 'CEBVD', 'CKD', 'COPD',
#             'CHF', 'CAD', 'DEPRN', 'DIABTYP1', 'DIABTYP2', 'DIABTYP0', 'ESRD',
#             'HPE', 'EDHIV', 'HYPLIPID', 'HTN', 'OBESITY', 'OSA', 'OSTPRSIS','SUBSTAB',]

# target_select = ['DOA','DIEDED','ADMIT']

# C_select = ['RFV1','RFV2','RFV3','RFV4','RFV5']
# datas['padding'] = ' '
# datas['CHIEFCOMPLAIN'] = datas['RFV1']+datas['padding']+datas['RFV2']+datas['padding']+datas['RFV3']+datas['padding']+datas['RFV4'].fillna('')+datas['padding']+datas['RFV5'].fillna('')
# datas['CHIEFCOMPLAIN'].str.replace('Blank','')

# ALL_SELECT = S_select+['CHIEFCOMPLAIN',]+H_select+target_select

# (datas['TEMPF'] - 32)/1.8
