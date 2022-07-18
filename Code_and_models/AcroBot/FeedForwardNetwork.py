# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:55:43 2022

@author: sarah
"""
import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_nodes: int):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden_nodes)
        self.l2 = nn.Linear(hidden_nodes, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x