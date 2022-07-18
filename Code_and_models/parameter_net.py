# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:07:59 2022

@author: sarah
"""


import torch.nn as nn


class parameter_network(nn.Module):
    def __init__(self, input_size, hidden_layer, output_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.l3 = nn.Linear(hidden_layer, output_dim)
        
    def clear_memory(self):
        pass

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x[0,:], x[1:,:] 
