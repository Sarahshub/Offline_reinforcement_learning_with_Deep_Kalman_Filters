# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:22:34 2022

@author: sarah
"""

import gym
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class IPolicy:
    
    def act(self, observation):
        """override method in subclass to implement policy"""
        pass
    
    def gain_insight(self, observation, reward, done, _info):
        """Override to use insight for each step"""
        pass


class Acrobot:
    """https://www.gymlibrary.ml/environments/classic_control/acrobot/?highlight=acrobot"""
    
    def __init__(self, policy: IPolicy, render: bool=True, environment_output: str='training_dir'):
        self.policy = policy
        self.env = gym.wrappers.Monitor(gym.make('Acrobot-v1'), environment_output, force=True)
        self.render = render
    
    
    def run_episode(self, number_of_steps: int) -> int:
        """Runs {number_of_steps} for a episode
        Returns total reward"""

        total_reward = 0

        observation = self.env.reset()
        for t in range(number_of_steps):
            action = self.policy.act(observation)
            
            (observation, reward, done, _info) = self.env.step(action)
            
            self.policy.gain_insight(observation, reward, done, _info)
            
            total_reward = total_reward + reward
            
            if self.render and t % 3 == 0: 
                self.env.render()
            if done:
                break
            
        return total_reward
    
    def __del__(self):
        self.env.render()
        self.env.close()
        
