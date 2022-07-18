# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:27:07 2022

@author: Sarah
""" 
import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
gc.collect()
torch.cuda.empty_cache()
#%%
EPISODES = 50  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 64  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
LOAD_MODEL = False # Load model to get expert data
SAVE_MODEL = False # Don't overwrite exsisting model  
SAVE_DATA = False # Decide to save memory buffer


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#%%
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.reward = []
        self.overall_reward = []
        self.episode = {}
        self.steps = {}
        
        

    def push(self, transition):
        self.memory.append(transition)
        self.reward.append(transition[3].item())
        return transition
        
        
    def add_to_episodes(self, n_step, actions):
        self.steps[n_step] = actions
        
    def summerzie_reward(self):
        episode_reward = sum(self.reward)
        self.overall_reward.append(episode_reward)
        self.reward = []
        return episode_reward
    
    
    def update(self):
        #self.episodes.append(self.steps)
        self.episode[e] = self.steps
        self.steps = {}
    
    def dict_of_episodes(self):
        return self.episode
        
    
    def total_reward(self):
        return self.overall_reward
    
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x
    
#%%
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '.', force=True)

model = None
if LOAD_MODEL == True:
    model = torch.load('model_b32_h64_LR0.001.pt')
    model.eval()
else: 
    model = Network()

if use_cuda:
    model.cuda()
memory = ReplayMemory(10000) #memory size 

optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []
#%%
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])



#%%
def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        environment.render()
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action[0, 0].item() )

        # negative reward when attempt ends
        if done:
            reward = -1

        stuff = memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))
        
        memory.add_to_episodes(steps, stuff)
        
        #print('steps ', steps, 'stuff ', stuff)
        
        learn()

        state = next_state
        steps += 1

        if done:
            memory.update()
            r = memory.summerzie_reward()
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            print(f'total reward in episode {r}')
            plot_durations()
            break
#%%

def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    #print('Batch state', batch_state, 'action', batch_action, 'batch_reward', batch_reward, 'next state', batch_next_state)

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    #print('Current q ', current_q_values)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    #print('max_next_q ', max_next_q_values)
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)
    #print('expected_q_values ', expected_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(-1,1)) 

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%%
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Trained model 10000 obs')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
#%%
e = 0
for e in range(EPISODES):
    run_episode(e, env)
    

    if e == EPISODES:
        dict_of_epi = memory.dict_of_episodes()
        all_reward = memory.total_reward()
        subset = all_reward[-99:]
        avg = sum(subset)/100
        print(f'Models average total reward for last 100 episodes {avg}, For hyperparam episode {EPISODES}, Discount factor {GAMMA}, Learning rate {LR}, hidden layers {HIDDEN_LAYER}, batch size {BATCH_SIZE}') 
        

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

#%%
if SAVE_MODEL == True:
    torch.save(model, 'model_tra191.49.pt')

if SAVE_DATA == True:
    torch.save(memory.dict_of_episodes, 'ReplayMemory_dict.pt')
    
    
    
    