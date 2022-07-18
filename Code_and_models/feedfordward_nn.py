# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:28:34 2022

@author: sarah
"""

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, hidden_nodes):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, hidden_nodes)
        self.l2 = nn.Linear(hidden_nodes, 2)
        
    def clear_memory(self):
        pass

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = x.squeeze()
        return x