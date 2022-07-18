# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:49:08 2022

@author: sarah
"""

import acrobot
import random
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import logging
import sys

# Setting logger
logging.basicConfig(stream = sys.stdout)
log = logging.getLogger("default")
log.setLevel(logging.INFO)

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class GreedyPolicy(acrobot.IPolicy):
    
    def __init__(self, 
                 model: nn.Module,
                 eps_start: float = 0.9, 
                 eps_end: float = 0.05, 
                 eps_decay: float = 200,
                 gamma: float = 0.8,
                 memory_sample_size: int = 500,
                 replay_batch_size: int = 16, 
                 learn = False):
        self.steps_done = 0
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.gamma = gamma
        
        self.replay_buffer = collections.deque(maxlen=memory_sample_size)
        self.last_observation = None
        self.last_action = None
        self.model = model
        
        self.replay_batch_size = replay_batch_size
        
        self.__action_choices__ = list(range(3))
        self.current_iteration = 0
        self.__history__ = {}
        self.learn = learn
    
    
    def act(self, observation):
        observation = Tensor(observation)

        self.last_observation = observation
        
        #sample = random.random()
        #eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        action = None
        #if sample > eps_threshold:
        action = self.model(observation.unsqueeze(0))
        if len(action.shape)>1:
            action = action.squeeze()
                
        action = action.argmax(0).item()
        #else:
            #action = random.sample(self.__action_choices__, 1)[0]
        
        self.last_action = action
        log.debug(f'Action: {action}')
        
        return action
    
    
    def gain_insight(self, observation, reward, done, _info):
        
        self.replay_buffer.append((
            self.last_observation,
            self.last_action,
            reward,
            Tensor(observation)
        ))
        
        self.__learn()
        
        if done:
            self.__history__[self.current_iteration] = list(self.replay_buffer)
            self.replay_buffer.clear()
            self.current_iteration += 1
            

    def __learn(self):
        if self.learn == False: 
            return
        if len(self.replay_buffer) < self.replay_batch_size:
            """Skip learning if less than 1 batch size recorded"""
            return
        
        samples = random.sample(list(self.replay_buffer), self.replay_batch_size)
        last_observation, action, reward, observation = zip(*samples)
        
        state = torch.stack(last_observation)
        action = LongTensor(action).unsqueeze(1)
        reward = Tensor(reward)
        next_state = torch.stack(observation)
    
        # current Q values are estimated by NN for all actions
        current_q_values = self.model(state).gather(1, action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(next_state).detach().max(1)[0]
        expected_q_values = reward + (self.gamma * max_next_q_values)
        
        
        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(-1,1)) 
        
        # backpropagation of loss to NN
        loss.backward()
        
    def save_history(self, filename: str):
        if filename[-3:] != ".pt":
            raise Exception(f'Filename must end with .pt, you gave: {filename}')
        torch.save(self.__history__, filename)
