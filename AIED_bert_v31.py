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

from transformers import BertModel, BertTokenizer, configuration_bert
import torch

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class bert_baseModel(nn.Module):
    def __init__(self, init_weight= None):
        super(bert_baseModel, self).__init__()    
        pretrained_weights = "bert-base-multilingual-cased"
        config = configuration_bert.BertConfig().from_pretrained(pretrained_weights) 
        try:
            self.bert_model = BertModel.from_pretrained(pretrained_weights)
            print('from_pretrained BERT ')
        except:
            self.bert_model = BertModel(config=config)
            print('No_pretrained BERT ')
        
    def forward(self, input_ids,attention_mask, position_ids=None,token_type_ids=None,train_mlm=False):    
        x = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        if train_mlm:
            return x[0][:,0,:], x[0]
        else:
            return x[0][:,0,:]

class bert_csp_train(nn.Module):
    def __init__(self):
        super(bert_csp_train, self).__init__()    
        pretrained_weights = "bert-base-multilingual-cased"
        config = configuration_bert.BertConfig().from_pretrained(pretrained_weights) 
        
        self.avgpool = nn.Sequential(nn.Linear(config.hidden_size,256),
                                     nn.GELU(),
                                     nn.BatchNorm1d(256),
                                     nn.Dropout(0.5),
                                     nn.Linear(256,64),
                                     nn.BatchNorm1d(64),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(64,2)
                                     )
    def forward(self, hidden_states):
        pooled_output = self.avgpool(hidden_states)
        return pooled_output

class bert_simCLR(nn.Module):
    def __init__(self):
        super(bert_simCLR, self).__init__()    
        pretrained_weights = "bert-base-multilingual-cased"
        config = configuration_bert.BertConfig().from_pretrained(pretrained_weights) 
        
        self.avgpool = nn.Sequential(nn.Linear(config.hidden_size,192),
                                     nn.GELU(),
                                     nn.BatchNorm1d(192),
                                     nn.Dropout(0.5),
                                     nn.Linear(192,config.hidden_size),
                                     nn.GELU(),
                                     nn.BatchNorm1d(config.hidden_size),
                                     )
    def forward(self, hidden_states):
        pooled_output = self.avgpool(hidden_states)
        return pooled_output
    
class BertLayerNorm(nn.Module):
    def __init__(self, config):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config['hidden_size']))
        self.bias = nn.Parameter(torch.zeros(config['hidden_size']))
        self.variance_epsilon = config['eps']

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class alphabetEmbedding(nn.Module):
    def __init__(self, config, init_weight= None):
        super(alphabetEmbedding, self).__init__()   
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, inputs_embeds=None, token_type_ids=None, position_ids=None, input_ids=None):
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class alphabetSelfAttention(nn.Module):
    def __init__(self, config):
        super(alphabetSelfAttention, self).__init__()

        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = context_layer
        return outputs

class alphabetSelfOutput(nn.Module):
    def __init__(self, config):
        super(alphabetSelfOutput, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class alphabetAttention(nn.Module):
    def __init__(self, config):
        super(alphabetAttention, self).__init__()
        self.self = alphabetSelfAttention(config)
        self.output = alphabetSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor,attention_mask):
        self_outputs = self.self(input_tensor,attention_mask)
        outputs = self.output(self_outputs, input_tensor)
        return outputs

class alphabetIntermediate(nn.Module):
    def __init__(self, config):
        super(alphabetIntermediate, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class alphabetOutput(nn.Module):
    def __init__(self, config):
        super(alphabetOutput, self).__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class alphabetLayer(nn.Module):
    def __init__(self, config):
        super(alphabetLayer, self).__init__()
        self.attention = alphabetAttention(config)
        self.intermediate = alphabetIntermediate(config)
        self.output = alphabetOutput(config)

    def forward(self, hidden_states,attention_mask):
        attention_output = self.attention(hidden_states,attention_mask)
        intermediate_output = self.intermediate(attention_output)
        outputs = self.output(intermediate_output, attention_output)
        return outputs    

class alphabetEncoder(nn.Module):
    def __init__(self, config):
        super(alphabetEncoder, self).__init__()
        self.layer = nn.ModuleList([alphabetLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states,attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states,attention_mask)

        outputs = hidden_states

        return outputs
    
class alphabetPooler(nn.Module):
    def __init__(self, config):
        super(alphabetPooler, self).__init__()
        self.avgpool = nn.Sequential(nn.Linear(config['hidden_size'],4*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4*config['hidden_size'],2*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config['hidden_size'],config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(config['hidden_size'],2)
                                     )
#        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.avgpool(hidden_states)
        return pooled_output

class alphaBertModel(nn.Module):
    r"""
 
    """
    def __init__(self, config):
        super(alphaBertModel, self).__init__()
        self.embeddings = alphabetEmbedding(config)
        self.encoder = alphabetEncoder(config)
        self.pooler = alphabetPooler(config)

    def forward(self, input_ids,attention_mask, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output,extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs)
        
        return pooled_output,encoder_outputs  # sequence_output, pooled_output, (hidden_states), (attentions)  
    
    
class structure_emb(nn.Module):
    def __init__(self, config):
        super(structure_emb, self).__init__()
        self.stc2emb = nn.Sequential(nn.Linear(config['structure_size'],16*config['structure_size']),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm1d(16*config['structure_size']),
                                     nn.Linear(16*config['structure_size'],8*config['structure_size']),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm1d(8*config['structure_size']),
                                     nn.Linear(8*config['structure_size'],config['hidden_size']),
                                     nn.LayerNorm(config['hidden_size']),
                                     )

    def forward(self, hidden_states):
        pooled_output = self.stc2emb(hidden_states)
        return pooled_output

class emb_emb(nn.Module):
    def __init__(self, config):
        super(emb_emb, self).__init__()
        self.emb_emb = nn.Sequential(nn.Linear(config['bert_hidden_size'],2*config['hidden_size']),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config['hidden_size'],config['hidden_size']),
                                     nn.LayerNorm(config['hidden_size']),
                                     )

    def forward(self, hidden_states):
        pooled_output = self.emb_emb(hidden_states)
        return pooled_output    
    
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None 
    
class GnLD(nn.Module):
    def __init__(self,config,device):
        super(GnLD, self).__init__()
        self.config = config
        self.device = device
        self.embeddings = alphabetEmbedding(config)
        self.encoder = alphabetEncoder(config)
        self.pooler = alphabetPooler(config)
        
    def forward(self, EDisease, M, SEP_emb_emb, nohx, token_type_ids=None, expand_data=None,mask_ratio=0.15,mask_ratio_pi=0.5,DS_model=None,mix_ratio=0.0,use_pi=False,yespi=None):
        bs = EDisease.shape[0]
        
        if use_pi:
            expand_data_sz = 1
            EM = torch.cat([M[:,:1],EDisease,SEP_emb_emb,M[:,1:]],dim=1)

            input_shape = EM.size()[:-1]
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)
                token_type_ids[:,2:] = 1

            attention_mask = torch.ones(EM.shape[:2],device=self.device)

            for i,e in enumerate(nohx):
                if e<2:
                    attention_mask[i,-1-expand_data_sz] = 0

                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0

            for i,e in enumerate(yespi):
                if e<1:
                    attention_mask[i,-1] = 0

                else:
                    rd = random.random()
                    if rd < mask_ratio_pi:
                        attention_mask[i,-1] = 0                        
                                      
        else:       
            expand_data_sz = 0
            if expand_data is None:
                EM = torch.cat([M[:,:1],EDisease,SEP_emb_emb,M[:,1:]],dim=1)
            else:
                EM = torch.cat([M[:,:1],EDisease,SEP_emb_emb,M[:,1:],expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]

            input_shape = EM.size()[:-1]
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)
                token_type_ids[:,2:] = 1

            attention_mask = torch.ones(EM.shape[:2],device=self.device)

            for i,e in enumerate(nohx):
                if e<2:
                    attention_mask[i,-1-expand_data_sz] = 0

                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0

            if expand_data is not None:
                attention_mask[:,-1*expand_data_sz:] = expand_data['mask']                

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 

        output = DS_model.embeddings(EM,token_type_ids)   

        if mix_ratio < 1:          
            output_e = DS_model.encoder(output,extended_attention_mask)
            output_d = self.encoder(output,extended_attention_mask)                      
            output = mix_ratio*output_d + (1-mix_ratio)*output_e                       
        else:
            output = self.encoder(output,extended_attention_mask)  

        output = DS_model.pooler(output[:,0]) 

        return output

class PriorD(nn.Module):
    def __init__(self,config,device):
        super(PriorD, self).__init__()
        self.config = config
        self.device = device
        self.dense = nn.Sequential(nn.Linear(config['hidden_size'],4*config['hidden_size']),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(4*config['hidden_size']),
                                   nn.Linear(4*config['hidden_size'],2*config['hidden_size']),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(2*config['hidden_size']),
                                   nn.Linear(2*config['hidden_size'],1),
                                   nn.Sigmoid()
                                   )  
        
    def forward(self, EDisease):
        output = self.dense(EDisease)
        
        return output
    
def target_real_fake(batch_size, device, soft):
    t = torch.ones(batch_size,1,device=device) 
    return soft*t, 1 - soft*t, t, 1-t
    
class DIM(nn.Module):
    def __init__(self,config,device='cpu',alpha=1, beta=1, gamma=10):
        super(DIM, self).__init__()
        self.config = config
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.GnLD = GnLD(config,device)
        self.PriorD = PriorD(config,device)
        
    def shuffleE(self,EDiseaseFake,bs):
        t = random.randint(1,200)
        for _ in range(t):
            r = random.random()
            s = random.randint(1,min(95,bs-1))
            if r >0.5:
                EDiseaseFake = torch.cat([EDiseaseFake[s:],EDiseaseFake[:s]],dim=0)
            else:
                EDiseaseFake = torch.cat([EDiseaseFake[:,s:],EDiseaseFake[:,:s]],dim=1)
        return EDiseaseFake
               
    def forward(self, 
                EDisease,
                M,
                SEP_emb_emb,
                nohx,
                token_type_ids=None,
                soft=0.7,
                DANN_alpha=1.2,
                expand_data=None,
                mask_ratio=0.15,
                mode=None,
                ptloss=False,
                DS_model=None,
                mix_ratio=0.0,
                mask_ratio_pi=0.5,
                EDisease2=None,
                shuffle=False,
                use_pi=False,
                yespi=None,
                ep=0):
        
        bs = EDisease.shape[0]
        EDiseaseFake = torch.cat([EDisease[1:],EDisease[:1]],dim=0)
 
        fake_domain, true_domain, fake_em, true_em = target_real_fake(batch_size=bs, device=self.device, soft=soft)
        
        criterion_DANN = nn.MSELoss().to(self.device)
        criterion_em = nn.CrossEntropyLoss().to(self.device)
        #using Transformer to similar Global + Local diversity
        
        if self.alpha ==0:
            GLD_loss = torch.tensor(0)            
            GLD0_loss = torch.tensor(0)
            GLD1_loss = torch.tensor(0)

        else:
            #GLD0 = -1*F.softplus(-1*self.GnLD(EDisease, M, SEP_emb_emb, token_type_ids=None)).mean()
            #GLD1 = -1*F.softplus(-1*self.GnLD(EDiseaseFake, M, SEP_emb_emb, token_type_ids=None)).mean()
            GLD0 = self.GnLD(EDisease, M, SEP_emb_emb, nohx, token_type_ids=None, expand_data=None,DS_model=DS_model,mix_ratio=mix_ratio,use_pi=use_pi,yespi=yespi)
            GLD1 = self.GnLD(EDiseaseFake, M, SEP_emb_emb, nohx, token_type_ids=None, expand_data=None,DS_model=DS_model,mix_ratio=mix_ratio,use_pi=use_pi,yespi=yespi)

            if shuffle:
                EDiseaseFake2 = self.shuffleE(EDisease,bs)   
                EDiseaseFake3 = self.shuffleE(EDisease2,bs)
                GLD2 = self.GnLD(EDiseaseFake2, M, SEP_emb_emb, nohx, token_type_ids=None, expand_data=None,DS_model=DS_model,mix_ratio=mix_ratio,use_pi=use_pi,yespi=yespi)
                GLD3 = self.GnLD(EDiseaseFake3, M, SEP_emb_emb, nohx, token_type_ids=None, expand_data=None,DS_model=DS_model,mix_ratio=mix_ratio,use_pi=use_pi,yespi=yespi)

                Contrast0 = torch.cat([GLD0[:,:1],GLD1[:,:1],GLD2[:,:1],GLD3[:,:1]],dim=-1)
                Contrast1 = torch.cat([GLD0[:,1:],GLD1[:,1:]],dim=-1)
                Contrast2 = torch.cat([GLD0[:,1:],GLD2[:,1:]],dim=-1)
                Contrast3 = torch.cat([GLD0[:,1:],GLD3[:,1:]],dim=-1)

                GLD0_loss = criterion_em(Contrast0,true_em.view(-1).long())
                GLD1_loss = criterion_em(Contrast1,fake_em.view(-1).long())
                GLD1_loss+= criterion_em(Contrast2,fake_em.view(-1).long()) 
                GLD1_loss+= criterion_em(Contrast3,fake_em.view(-1).long()) 
            else:
                Contrast0 = torch.cat([GLD0[:,:1],GLD1[:,:1]],dim=-1)
                Contrast1 = torch.cat([GLD0[:,1:],GLD1[:,1:]],dim=-1)

                GLD0_loss = criterion_em(Contrast0,true_em.view(-1).long())
                GLD1_loss = criterion_em(Contrast1,fake_em.view(-1).long())

            GLD_loss = self.alpha*(GLD0_loss+GLD1_loss)
        
        #SimCLR
        if self.beta ==0:
            loss_simCLR = torch.tensor(0)
        else:
            if EDisease2 is not None:
                Eall = torch.cat([EDisease.view(bs,-1),EDisease2.view(bs,-1)],dim=0)
                nEall = F.normalize(Eall,dim=1)
                simCLR = (1.+ep/150)*torch.mm(nEall,nEall.T)
                simCLR = simCLR - 1e3*torch.eye(simCLR.shape[0],device=self.device)

                simtrg= torch.arange(2*bs,dtype=torch.long,device=self.device)
                simtrg = torch.cat([simtrg[bs:],simtrg[:bs]])

                loss_simCLR = self.beta*criterion_em(simCLR,simtrg)
                           
        #using GAN method for train prior       
        fake_domain+=(1.1*(1-soft)*torch.rand_like(fake_domain,device=self.device))
        true_domain-=(1.1*(1-soft)*torch.rand_like(true_domain,device=self.device))        
             
        # Proir setting
        '''
        owing to y= x ln x, convex function, a+b+c=1; a,b,c>0, <=1; when a=b=c=1/3, get the min xln x
        set prior=[-1,1] uniform
        '''
        prior = torch.rand_like(EDisease.view(bs,-1),device=self.device)
        #prior = (prior - prior.mean())/(prior.std()+1e-6)   #fit to Domain of Layernorm()
        prior = 2*prior-1                                   #fit to Domain of Tanh()
        
        if self.gamma ==0:
            prior_loss = torch.tensor(0)
        else:
            if mode=='D':
                #only train the D , not G
                for param in self.PriorD.parameters():
                    param.requires_grad = True
                #d_EDisease = EDisease.view(bs,-1).detach()
                pred_domain_T = self.PriorD(EDisease.view(bs,-1).detach())
                loss_domain_T = criterion_DANN(pred_domain_T,true_domain)
                pred_domain_F = self.PriorD(prior) 
                loss_domain_F = criterion_DANN(pred_domain_F,fake_domain)          
            elif mode=='G':
                #only train the G , not D
                for param in self.PriorD.parameters():
                    param.requires_grad = False

                pred_domain_T = self.PriorD(EDisease.view(bs,-1))
                loss_domain_T = criterion_DANN(pred_domain_T,fake_domain)
                loss_domain_F = 0

            prior_loss = self.gamma*(loss_domain_T+loss_domain_F)
        
        if ptloss:
            with torch.no_grad():
                if EDisease2 is None:
                    print('GT:{:.4f}, GF:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                                  GLD1_loss.item(),
                                                                                  prior_loss.item()
                                                                                  ))
                else:
                    print('GT:{:.4f}, GF:{:.4f}, Sim:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                                 GLD1_loss.item(),
                                                                                 loss_simCLR.item(),
                                                                                 prior_loss.item()))          
 
                print(EDisease[0,0,:24])
                print(EDisease[1,0,:24])
                if self.alpha >0:                 
                    print('GLD0',GLD0[:2])#,true_em,true_domain)
                    print('GLD1',GLD1[:2])#,fake_em,fake_domain)
                    print('Cts0',Contrast0[:2])#,true_em,true_domain)
                    print('Cts1',Contrast1[:2])#,fake_em,fake_domain)
                if self.beta >0:
                    print('Sim',simCLR[bs-1:bs+5,:8])
                    print('Strg',simtrg[bs-4:bs+8])
                
        if EDisease2 is None:
            return GLD_loss+prior_loss
        else:
            return GLD_loss+prior_loss+loss_simCLR

class EDis(nn.Module):
    def __init__(self, config):
        super(EDis, self).__init__()
        self.EDisemb = nn.Sequential(nn.Linear(config['hidden_size'],config['hidden_size']),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(config['hidden_size'],config['hidden_size']),
                                     nn.Tanh(),
                                     )

    def forward(self, hidden_states):
        ED = self.EDisemb(hidden_states.squeeze(1))
        return ED.unsqueeze(1)  
    
class ewed_Model(nn.Module):
    def __init__(self,config,tokanizer,device='cpu',fixpretrain=True):
        super(ewed_Model, self).__init__() 
        self.config = config
        #self.baseBERT = bert_baseModel()
        self.stc2emb = structure_emb(config)
        self.emb_emb = emb_emb(config)                        
        
        self.embeddings = alphabetEmbedding(config)
        self.encoder = alphabetEncoder(config)
        self.pooler = alphabetPooler(config)
        
        self.EDis = EDis(config)
        self.tokanizer = tokanizer
        self.device = device
        
        '''
        if fixpretrain:
            for param in self.baseBERT.parameters():
                param.requires_grad = False
        '''
    def forward(self,
                baseBERT,
                inputs,
                normalization=None, 
                noise_scale=0.001,
                mask_ratio=0.15, 
                mask_ratio_pi=0.5,
                token_type_ids=None, 
                expand_data=None,
                use_pi=False,
                test=False):
               
        s,c,cm,h,hm = inputs['structure'],inputs['cc'],inputs['mask_cc'],inputs['ehx'],inputs['mask_ehx']
        s,c,cm,h,hm = s.to(self.device),c.to(self.device),cm.to(self.device),h.to(self.device),hm.to(self.device)
               
        if normalization is None:
            s_noise = s
        else:
            #normalization = torch.tensor(normalization).expand(s.shape).to(self.device)
            normalization = torch.ones(s.shape).to(self.device)
            noise_ = normalization*noise_scale*torch.randn_like(s,device=self.device)
            s_noise = s+noise_
            
        baseBERT.eval()
        s_emb = self.stc2emb(s_noise)
        s_emb_org = self.stc2emb(s)
        c_emb = baseBERT(c.long(),cm.long())
        h_emb = baseBERT(h.long(),hm.long())
        
        CLS_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.cls_token_id],device=self.device))
        SEP_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.sep_token_id],device=self.device))
        PAD_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.pad_token_id],device=self.device))
                
        cumsum_hx_n = torch.cumsum(inputs['stack_hx_n'],0)
        h_emb_mean_ = []
        for i,e in enumerate(cumsum_hx_n):            
            if inputs['stack_hx_n'][i]>1:
                h_mean = torch.mean(h_emb[1:cumsum_hx_n[i]],dim=0) if i < 1 else torch.mean(h_emb[1+cumsum_hx_n[i-1]:cumsum_hx_n[i]],dim=0)
                h_emb_mean_.append(h_mean)
            else:
                h_emb_mean_.append(PAD_emb.view(h_emb[0].shape))
                
        h_emb_mean = torch.stack(h_emb_mean_)
        h_emb_mean.to(self.device)
               
        c_emb_emb = self.emb_emb(c_emb)
        h_emb_emb = self.emb_emb(h_emb_mean)
               
        CLS_emb_emb = self.emb_emb(CLS_emb)
        SEP_emb_emb = self.emb_emb(SEP_emb)
        
        CLS_emb_emb = CLS_emb_emb.expand(c_emb_emb.shape)
        SEP_emb_emb = SEP_emb_emb.expand(c_emb_emb.shape)

        CLS_emb_emb.unsqueeze_(1)
        SEP_emb_emb.unsqueeze_(1)

        if use_pi:
            pi,pm,pil,yespi = inputs['pi'],inputs['mask_pi'],inputs['origin_pi_length'],inputs['yesPI']
            pi,pm,pil,yespi = pi.to(self.device),pm.to(self.device),pil.to(self.device),yespi.to(self.device)       
            p_emb = baseBERT(pi.long(),pm.long())
            
            p_emb_emb = self.emb_emb(p_emb)

            expand_data_sz = 1
            input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            
            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0            
            
            nopi = inputs['yesPI'] < 1
            for i,e in enumerate(nopi):
                if e:
                    attention_mask[i,-1] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio_pi:
                        attention_mask[i,-1] = 0                               
        else:  
            p_emb = None
            yespi = None
            expand_data_sz = 0
            if expand_data is None:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
            else:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]

            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    if test:
                        pass
                    else:
                        rd = random.random()
                        if rd < mask_ratio:
                            attention_mask[i,-1-expand_data_sz] = 0
            if expand_data is not None:
                attention_mask[:,-1*expand_data_sz:] = expand_data['mask']

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   

        output = self.embeddings(input_emb,token_type_ids)
        output = self.encoder(output,extended_attention_mask)

        EDisease = output[:,:1]
        #g_class = self.gen_class(ptemb)
        #g_decoder = self.gen_decoder(ptemb)

        output = self.EDis(EDisease)

        return output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), inputs['stack_hx_n'],expand_data

    
class ewed_expand_Model(nn.Module):
    def __init__(self,config):
        super(ewed_expand_Model, self).__init__() 
        self.config = config
        
    def forward(self, inputs, emb_model): 
        self.device = emb_model.device
        pi,pm,pil,yespi = inputs['pi'],inputs['mask_pi'],inputs['origin_pi_length'],inputs['yesPI']
        pi,pm,pil,yespi = pi.to(self.device),pm.to(self.device),pil.to(self.device),yespi.to(self.device)       

        p_emb = emb_model.baseBERT(pi.long(),pm.long())                         
        p_emb_emb = emb_model.emb_emb(p_emb)                       
        expand_emb = p_emb_emb.unsqueeze_(1)

        attention_mask = torch.ones(expand_emb.shape[:2],device=self.device)
        for i,e in enumerate(yespi):
            if e<1:
                attention_mask[i,-1] = 0  
        
        output = {'emb':expand_emb,
                  'mask':attention_mask}     
                      
        return output  

class cls_model(nn.Module):
    def __init__(self,config):
        super(cls_model, self).__init__() 
        self.config = config
        self.hidden0 =   nn.Sequential(nn.Linear(config['hidden_size'],4*config['hidden_size']),
                                       nn.GELU(),
                                       nn.Dropout(0.5),
                                       nn.BatchNorm1d(4*config['hidden_size']),
                                       nn.Linear(4*config['hidden_size'],1*config['hidden_size']),
                                       nn.GELU(),
                                       nn.Dropout(0.5),
                                       nn.BatchNorm1d(1*config['hidden_size']),
                                       )
        self.hidden1 =   nn.Sequential(nn.Linear(config['hidden_size'],4*config['hidden_size']),
                                       nn.GELU(),
                                       nn.Dropout(0.5),
                                       nn.BatchNorm1d(4*config['hidden_size']),
                                       nn.Linear(4*config['hidden_size'],2*config['hidden_size']),
                                       nn.GELU(),
                                       nn.Dropout(0.5),
                                       nn.BatchNorm1d(2*config['hidden_size']),                                      
                                       nn.Linear(2*config['hidden_size'],2),
                                       )
    def forward(self, EDisease):
        resout = EDisease
        for i in range(3):
            output = self.hidden0(resout)
            resout = resout+output
        output = self.hidden1(resout)
        return output
                
class ewed_CLS_Model(nn.Module):
    def __init__(self,config,device):
        super(ewed_CLS_Model, self).__init__() 
        self.config = config      
        self.dense_icu = cls_model(config)
        self.dense_tri = cls_model(config)      
        self.dense_die = cls_model(config)
        self.dense_poor = cls_model(config)
        self.device = device
        
    def forward(self, EDisease):
        bs = EDisease.shape[0]
        cls_icu = self.dense_icu(EDisease.view(bs,-1)) 
        cls_die = self.dense_die(EDisease.view(bs,-1)) 
        cls_tri = self.dense_tri(EDisease.view(bs,-1))
        cls_poor = self.dense_poor(EDisease.view(bs,-1))
        
        return cls_icu,cls_die,cls_tri,cls_poor
    
class pickle_Model(nn.Module):
    def __init__(self,config,tokanizer,device='cpu',fixpretrain=True):
        super(pickle_Model, self).__init__() 
        self.config = config
        self.stc2emb = structure_emb(config)
        self.emb_emb = emb_emb(config)                        
        
        self.embeddings = alphabetEmbedding(config)
        self.encoder = alphabetEncoder(config)
        self.pooler = alphabetPooler(config)
        
        self.EDis = EDis(config)
        self.tokanizer = tokanizer
        self.device = device
        
    def forward(self,
                baseBERT,
                inputs,
                normalization=None, 
                noise_scale=0.001,
                mask_ratio=0.15, 
                mask_ratio_pi=0.5,
                token_type_ids=None, 
                expand_data=None,
                use_pi=False):
        s,c_emb_,h_emb_mean_ = inputs['structure'],inputs['ccemb'],inputs['hxemb']
        s,c_emb_,h_emb_mean_ = s.to(self.device),c_emb_.to(self.device),h_emb_mean_.to(self.device)
               
        normalization = torch.ones(s.shape).to(self.device)
        noise_ = normalization*noise_scale*torch.randn_like(s,device=self.device)
        s_noise = s+noise_   
#         s_noise = nn.Dropout(0.1*random.random())(s_noise)
            
        baseBERT.eval()
        s_emb = self.stc2emb(s_noise)
        s_emb_org = self.stc2emb(s)
        
        CLS_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.cls_token_id],device=self.device))
        SEP_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.sep_token_id],device=self.device))
        PAD_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.pad_token_id],device=self.device))

        normalization_c = torch.ones(c_emb_.shape).to(self.device)
        cc_noise_ = normalization_c*0.0*noise_scale*torch.randn_like(c_emb_,device=self.device)
        c_emb = c_emb_+cc_noise_  
#         c_emb = nn.Dropout(0.01*random.random())(c_emb)

        normalization_h = torch.ones(h_emb_mean_.shape).to(self.device)
        hx_noise_ = normalization_h*0.0*noise_scale*torch.randn_like(h_emb_mean_,device=self.device)
        h_emb_mean = h_emb_mean_+hx_noise_
#         h_emb_mean = nn.Dropout(0.01*random.random())(h_emb_mean)        

        c_emb_emb = self.emb_emb(c_emb)
        h_emb_emb = self.emb_emb(h_emb_mean)

        CLS_emb_emb = self.emb_emb(CLS_emb)
        SEP_emb_emb = self.emb_emb(SEP_emb)

        CLS_emb_emb = CLS_emb_emb.expand(c_emb_emb.shape)
        SEP_emb_emb = SEP_emb_emb.expand(c_emb_emb.shape)

        CLS_emb_emb.unsqueeze_(1)
        SEP_emb_emb.unsqueeze_(1)        
        
        if use_pi:
            p_emb_ = inputs['piemb']
            p_emb_ = p_emb_.to(self.device)
            
            normalization_p = torch.ones(p_emb_.shape).to(self.device)
            pi_noise_ = normalization_p*0.1*noise_scale*torch.randn_like(p_emb_,device=self.device)
            p_emb = p_emb_+pi_noise_  
            p_emb = nn.Dropout(0.1)(p_emb) 
            
            p_emb_emb = self.emb_emb(p_emb)
            
            expand_data_sz = 1
            
            input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
           
            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0
            pi_num = inputs['pi_num']
            nopi = pi_num < 1            
            for i,e in enumerate(nopi):            
                if e:
                    attention_mask[:,-1] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio_pi:
                        attention_mask[i,-1] = 0                    
       
        else:
            p_emb = None
            pi_num = None
            expand_data_sz = 0
            if expand_data is None:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
            else:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]

            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0
            if expand_data is not None:
                attention_mask[:,-1*expand_data_sz:] = expand_data['mask']

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   

        output = self.embeddings(input_emb,token_type_ids)
        output = self.encoder(output,extended_attention_mask)

        EDisease = output[:,:1]

        output = self.EDis(EDisease)
        return output,EDisease,(s,input_emb,input_emb_org),(CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,pi_num), inputs['stack_hx_n'],expand_data 
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

if __name__ == '__main__':
    import AIED_dataloader_v02
    BERT_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    all_datas = AIED_dataloader_v02.load_datas()

    data15_triage_train = all_datas['data15_triage_train']
    data01_person = all_datas['data01_person']
    data02_wh = all_datas['data02_wh']
    data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']
    data15_triage_train = all_datas['data15_triage_train']

    EDEW_DS = AIED_dataloader_v02.EDEW_Dataset(ds=data15_triage_train,
                           tokanizer = BERT_tokenizer,
                           data01_person = data01_person,
                           data02_wh = data02_wh,
                           data25_diagnoses= data25_diagnoses,
                           normalization = dm_normalization_np, 
                          )

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                             shuffle = True,
                             num_workers=8,
                             batch_size=8,
                             collate_fn=AIED_dataloader_v02.collate_fn)

    
    config = {'hidden_size': 96,
              'bert_hidden_size': 768,
              'max_position_embeddings':512,
              'eps': 1e-12,
              'input_size': 64,
              'vocab_size':64,
              'type_vocab_size':4,
              'hidden_dropout_prob': 0.1,
              'num_attention_heads': 8, 
              'attention_probs_dropout_prob': 0.2,
              'intermediate_size': 64,
              'num_hidden_layers': 8,
              'structure_size':12,
              'order_size':256
             }
    
    device = 'cuda'
    test_model = ewed_Model(config=config,
                            tokanizer=BERT_tokenizer,
                            device=device)
    
    test_model.to(device)
    
    
    
    for batch_idx, sample in enumerate(EDEW_DL):                  
        y = test_model(sample)
        print(batch_idx,)

        if batch_idx > 2:
            break
