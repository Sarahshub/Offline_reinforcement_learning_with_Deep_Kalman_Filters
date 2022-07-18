# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:39:16 2022

@author: 44791
"""
import torch
import torch.nn as nn



class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        #self.n = nn.BatchNorm2d(16)
        self.l4 = nn.Linear(8, 2)
        
    def clear_memory(self):
        pass

    def forward(self, x):
        x = torch.relu(self.l1(x))# stick with relu 
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        #x = nn.BatchNorm2d(x)
        x = self.l4(x)
        return x
    
#%%
class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, 64)
        self.l2 = nn.Linear(64, 2)
        
    def clear_memory(self):
        pass

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

#%%

#%% 
