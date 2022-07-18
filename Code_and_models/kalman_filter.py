# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:46:41 2022

@author: sarah
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import math
from parameter_net import parameter_network

#%%
# TODO: Do a proper hyperparameter tuning on the prior precision and the initialization of the parameters
class Kalman_filter(nn.Module):
    def __init__(self, n_dim, hidden_layer, output_dim):
        super(Kalman_filter, self).__init__()
        self.w = nn.Parameter(th.Tensor(n_dim))
        self.Q = Parameter(th.Tensor(n_dim, n_dim))
        self.fc = nn.Linear(n_dim, output_dim)
        self.mu_w = Parameter(th.Tensor(1, n_dim))
        self.bias = nn.Parameter(th.Tensor(n_dim))
        self.reset_parameters()
        self.p_model = parameter_network(n_dim, hidden_layer, n_dim)
        
        
    def clear_memory(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.w.data.normal_(0, stdv)
        self.Q.data.normal_(0, stdv) 
        self.bias.data.zero_()

    def forward(self, T): # feed e.g. 5 timepoints back skal vaere T til for loop
        #for t in range(st.shape[1]): 
        out = None
        for st in T:
            w = self.w.unsqueeze(0)
            e = th.randn_like(self.w) 
            Q = self.Q
            wQ=th.cat((w, Q),0)
            wd, dQ = self.p_model(wQ)
            w = w + wd
            Q = Q + dQ
            #print(f' w: {w}, shape {w.shape}, wt: {wt}, shape {wt.shape}, St {st} shape {st.shape} ')
            #print(f' Q: {Q}, shape {Q.shape}, Qt: {Qt}, shape {Qt.shape}')
            #print(f' w {self.w}, st {st}, Q {self.Q}, e {e} ')
            #print(f' w {self.w.unsqueeze(-1).shape}, st {st.shape}, Q {self.Q.shape}, e {e.shape} ')
            #print(f' wt shape: {wt.unsqueeze(0).shape}, st shape: {st.unsqueeze(-1).shape}')
            p_wst = th.mm(w.squeeze(1), st.unsqueeze(-1))
            #print(f'W st {p_wst.shape}')
            p_Qe = th.mm(Q, e.unsqueeze(-1))
            #print(f'Q e {p_Qe.shape}')
            st_t = p_wst + p_Qe
            #print(f'stt {st_t.shape}')
            out = st_t.view(-1)
        out = self.fc(out)
            #print(f'out {out}')
            #print(f'out-1 {out[-1]}')
        return out # action 
        # check dim for transpose
        # adjust according to sketch 

           # print(f'current st: {st}, predicted st: {st_t}')

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ' -> ' \
               + str(self.n_out) + ')'
               

