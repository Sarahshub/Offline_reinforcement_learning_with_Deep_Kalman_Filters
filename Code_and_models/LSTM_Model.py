# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:10:45 2022

@author: sarah
"""
import torch
import torch.nn as nn

from collections import deque

use_cuda = torch.cuda.is_available()

class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_nodes, layer_dim, output_dim):
        super(RNNNetwork, self).__init__()
        
        self.hidden_nodes = hidden_nodes
        self.input_size = input_size
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        

        self.Encoder = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_nodes, 
            num_layers = layer_dim
        )
        
        self.Decoder = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_nodes, 
            num_layers = layer_dim
        ) 
        self.fc = nn.Linear(hidden_nodes, output_dim)
        
    
    def clear_memory(self):
        pass

    def init_hidden(self) -> None:
        # the weights are of the form (layers, batches, hidden_nodes)
        hidden_a = torch.randn(self.layer_dim, 1, self.hidden_nodes)
        hidden_b = torch.randn(self.layer_dim, 1, self.hidden_nodes)

        if use_cuda:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        self.hidden = (hidden_a, hidden_b)
    
    # Expected x to be 1 observation of state, ie. shape(4)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        self.init_hidden()
        out, self.hidden = self.Encoder(x, self.hidden)
     
        out = out.reshape(-1, self.hidden_nodes)
        
        out = out[-1,]
        # reshaping (1,S,N) to (S,N)
        # S: sequence size
        # N: Hidden nodes
        out = self.fc(out) #Solve ODE for action
        return out