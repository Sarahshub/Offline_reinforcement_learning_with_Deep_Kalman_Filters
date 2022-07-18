# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:55:16 2022

@author: sarah
"""

import GreedyPolicy as gp
import acrobot
import FeedForwardNetwork as ffn
import logging
import sys
import torch
from kalman_filter import Kalman_filter

load_model = True
# Setting logger
logging.basicConfig(stream = sys.stdout)
log = logging.getLogger("default")
log.setLevel(logging.INFO)

use_cuda = torch.cuda.is_available() 
model = None
if load_model == True:
    model = torch.load('../DKF_AB.pt') # E3, HN10, MS 32
    model.eval()
else:
    model = ffn.FeedForwardNetwork(input_size=6, output_size=3, hidden_nodes=32)
model = model.cuda() if use_cuda else model

policy = gp.GreedyPolicy(model = model, gamma = 0.4)
acrobot_env = acrobot.Acrobot(policy = policy, render = True)


episodes = 1000
steps_per_episode = 500

for i in range(episodes):
    score = acrobot_env.run_episode(steps_per_episode)
    log.info(f'iteration: {i}, score: {score}')

filename = f'test_kalman_online.pt'
policy.save_history(filename = filename)
log.info(f'Saved results to: {filename}')


#torch.save(model, 'model_acrobot_below_200.pth')
# Cleans up window
del(acrobot_env)

#
#%%
import numpy as np
total = 0
l = []
for value in policy.__history__.values():
    episode_total = 0
    for v2 in value:
        total = total + v2[2]
        episode_total = episode_total + v2[2]
    l.append(episode_total)
    
