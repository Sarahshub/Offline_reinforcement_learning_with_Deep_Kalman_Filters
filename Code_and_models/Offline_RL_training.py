# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:39:40 2022

@author: sarah
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from torch.functional import F
import gc
import logging
import sys
n_choices = 3 #Discrete labels  

#%% Global config
gc.collect()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# Setting logger
log = logging.getLogger("default")

#%% Setup Torch for GPU/CPU
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#%% Load Class ReplayMemory

class ReplayMemory:
    def __init__(self, capacity, batch_size, data):
        self.capacity = capacity
        self.data_size = len(data)
        self.dataloader = None
        self.data = data
        self.current_episode = -1
        self.episode = None
        self.reward = []
        self.overall_reward = []
        self.loss = []
        self.overall_loss = []
        self.batch_acc = []
        self.overall_acc = []
        self.batch_size = batch_size
        self.val_pred_actual = []
        
    def summerzie_reward(self):
        episode_reward = sum(self.reward)
        self.overall_reward.append(episode_reward)
        self.reward = []
        return episode_reward
    
    def total_reward(self):
        return self.overall_reward
    
    def ECE_data(self, data):
        return self.val_pred_actual.append(data)
    
    
    def loss_t(self, l):
        self.loss.append(l.item())
        
    def summerize_loss(self):
        epoch_loss = sum(self.loss) / len(self.loss) if len(self.loss) else 0
        self.overall_loss.append(epoch_loss)
        self.loss = []
        return epoch_loss
    
    def save_acc(self, l):
        self.batch_acc.append(l.item())
        
    def epoch_acc(self):
        epoch_acc = ((sum(self.batch_acc) / len(self.batch_acc)) * 100) if len(self.batch_acc) else 0
        self.overall_acc.append(epoch_acc)
        self.batch_acc = []
        return epoch_acc
        
    def start_next_episode(self):
        self.current_episode = self.current_episode + 1
        self.episode = self.data[self.current_episode]
        self.dataloader = torch.utils.data.DataLoader(
            self.episode, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        return self.iterator()
    
    def reset_episodes(self):
        self.current_episode = -1
        self.episode = None
        self.dataloader = None
    
    def has_next_episode(self):
        return len(self.data) > self.current_episode + 1
        
    def iterator(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.data_size)





#%% Load class PerformanceMonitor

class PerformanceMonitor:
    def __init__(self):
        self.accuracies = {}
        self.losses = {}
        self.target = {}
        self.predictions = {}
       
    
    # group is current epoch where the result is logged
    def log_accuracy(self, group, accuracy):
        self._add_to_list_in_dict(self.accuracies, group, accuracy)
        
    
    # group is current epoch where the result is logged
    def log_loss(self, group, loss):
        self._add_to_list_in_dict(self.losses, group, loss)


    def get_accuracies(self, group=None):
        return self._retrive_logs(self.accuracies, group)


    def get_losses(self, group=None):
        return self._retrive_logs(self.losses, group)

    
    def _add_to_list_in_dict(self, dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = []
        dictionary[key].append(value)
        
    
    def _retrive_logs(self, dictionary, group):
        if group is None:
            return dictionary
        if group not in dictionary:
            raise Exception(f'No group: "{group}" found in performance logs, got: {list(dictionary.keys())}', group)
        return dictionary[group]
    
    def add_predict_target(self, pred, target, group):
        self._add_to_list_in_dict(self.target, group, target)
        self._add_to_list_in_dict(self.predictions, group, pred)
    
    
 
#%% Encode true labels
def classify_action(action, n_choices=n_choices):
    return F.one_hot(action, n_choices)
    
#%%
def run_epoch(epoch, monitor, model, optimizer, dataset, func, memory_size):
    total_episodes = len(dataset)
    print(f'episodes: {total_episodes}')
    current_episode = 0
    for episode in dataset:
        current_episode = current_episode + 1
        #if current_episode == 0:
            #print(f"Epoch: {epoch:03} Running episode: {current_episode:03}/{total_episodes:03}")
        #else:
            #print(f"\rEpoch: {epoch:03} Running episode: {current_episode:03}/{total_episodes:03}")
            
        model.clear_memory()
        rounds = len(episode) - memory_size
        if rounds <= 0: 
            continue
        
        for i in range(rounds):
            func(episode[i:i + memory_size], monitor, epoch, model, optimizer)

#%% Training
def learn(observations, monitor, epoch, model, optimizer):
    
    model.train()
    # random transition batch is taken from experience replay memory

    batch_state, batch_action, batch_next_state, batch_reward = zip(*observations)

    state = torch.stack(batch_state)
    
    last_action = batch_action[-1]
    action = last_action.long() if torch.is_tensor(last_action) else LongTensor([last_action])
    #state = batch_state.view(-1, 4)
    #action = batch_action.view(-1) #.float()
    
    optimizer.zero_grad()
    result = model(state)
    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    classifier = classify_action(action).float()

    loss = loss_fn(result.view(1,-1), classifier.view(1,-1))
    monitor.log_loss(group=epoch, loss=loss.item())

    loss.backward()#retain_graph=True)
    optimizer.step()
    
    models_action = result.argmax(dim = 0)
    accuracy = models_action.eq(action.flatten())

    monitor.log_accuracy(group=epoch, accuracy=accuracy.item())
    monitor.add_predict_target(group=epoch, target=action.item(), pred=models_action.item())
    
#%% validation
def validation(observations, monitor, epoch, model, optimizer):

     model.eval()
     with torch.no_grad():
         batch_state, batch_action, batch_next_state, batch_reward = zip(*observations)

         state = torch.stack(batch_state)
         
         last_action = batch_action[-1]
         action = last_action.long() if torch.is_tensor(last_action) else LongTensor([last_action])
         #state = batch_state.view(-1, 4)
         #action = batch_action.view(-1) #.float()
         
         result = model(state)
         #loss_fn = torch.nn.MSELoss()
         loss_fn = torch.nn.CrossEntropyLoss()
         
         classifier = classify_action(action).float()

         loss = loss_fn(result.view(1,-1), classifier.view(1,-1))
         monitor.log_loss(group=epoch, loss=loss.item())
         
         models_action = result.argmax(dim = 0)
         accuracy = models_action.eq(action.flatten())

         monitor.log_accuracy(group=epoch, accuracy=accuracy.item())
         monitor.add_predict_target(group=epoch, target=action.item(), pred=models_action.item())
         
         


#%% load data
if use_cuda:
    env = torch.load('training_data_800_q_learning_acro.pt')
else:
    env = torch.load('training_data_800_q_learning_acro.pt', map_location=torch.device('cpu'))


#%% unpack data

data = []
for k,v in env.items():
    data.append(list(v))

#%% Load parameters
#model_type = 'Kalman' # choose between FF for feed forward network, LSTM for RNN LSTM, Kalman for the kalman filter 
EPOCH = 3 # number of episodes
#LR = 0.001  # NN optimizer learning ratee
#WEGHT_DECAY = 0.000
LOAD_MODEL = False # Load model to get expert data
SAVE_MODEL = False # Don't overwrite exsisting model  
INPUT_SIZE = 6
#hidden_nodes = 32
layer_dimension = 2
output_dimension = 3
#MEMORY_SIZE = 16



#%% Setup training

# 80/20 split
#data = data[1:100]
train_size = int(0.7 * len(data))
val_size = len(data) - train_size

train, test,  = torch.utils.data.random_split(data, [train_size, val_size])

# UNUSED
# Setup replay memory
#memory_train = ReplayMemory(5000, BATCH_SIZE, train)
#memory_val = ReplayMemory(5000, BATCH_SIZE, train)




#%% Setup Model

from kalman_filter import Kalman_filter
from LSTM_Model import RNNNetwork
from feedfordward_nn import Network

def get_model(model_type, learning_rate, hidden_nodes, layers=None):
    input_model = None
    if model_type == 'LSTM':
            input_model = RNNNetwork(
                input_size = INPUT_SIZE, 
                hidden_nodes = hidden_nodes, 
                layer_dim = layer_dimension, 
                output_dim = output_dimension)
        
    elif model_type == 'Kalman':
            input_model = Kalman_filter(INPUT_SIZE, hidden_nodes, output_dimension)
    
    elif model_type == 'FF':
            input_model = Network(hidden_nodes = hidden_nodes)

    if use_cuda:
        input_model.cuda()
    optimizer = optim.Adam(input_model.parameters(), lr = learning_rate)
    
    return (input_model, optimizer)

  
#%% Running Epochs

def run_epochs(n_epochs, model, optimizer, train_data, val_data, memory_size=None):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    global validation_monitor
    global training_monitor
    
    validation_monitor = PerformanceMonitor()
    training_monitor = PerformanceMonitor()
    
    for epoch in range(n_epochs):
        log.info('===============================')
        log.info(f'Running Epoch {epoch}')
        
        log.info('== Running training for Epoch ==')
        run_epoch(epoch, training_monitor, model, optimizer, train_data, learn, memory_size)
        
        log.info('== Running validation for Epoch ==')
        run_epoch(epoch, validation_monitor, model, optimizer, val_data, validation, memory_size)
        
        e_loss = Tensor(training_monitor.get_losses(group=epoch)).mean().item()
        log.info(f'traning loss per epoch: {e_loss}')
        train_loss.append(e_loss)
        
        e_loss_val = Tensor(validation_monitor.get_losses(group=epoch)).mean().item()
        log.info(f'validation loss per epoch: {e_loss_val}')
        val_loss.append(e_loss_val)
        
        e_acc_train = Tensor(training_monitor.get_accuracies(group=epoch)).mean().item()
        log.info(f'Training acc per epoch: {e_acc_train}')
        train_acc.append(e_acc_train)
        
        e_acc = Tensor(validation_monitor.get_accuracies(group=epoch)).mean().item()
        log.info(f'Validation acc per epoch: {e_acc}')
        val_acc.append(e_acc)
        
        
    
    
    log.info(f'All {EPOCH} epochs complete')
    
    return (train_loss, train_acc, val_loss, val_acc, validation_monitor)

#%%

def make_plot(n_epochs, label_type, filename, *ys, save=False):
    epochs = range(1, n_epochs + 1)
    
    colors = deque(['g','b','y'], 10)
    
    for y in ys:
        data_source, data = y
        line_color = colors.popleft()
        plt.plot(epochs, data, line_color, label=f'{data_source} {label_type}')
    plt.title(f'Kalman {n_epochs} Training {label_type}')
    plt.xlabel('Epochs')
    plt.ylabel(label_type)
    plt.legend()
    if save:
        plt.savefig(filename)
    else: 
        plt.show()
    plt.clf()
    

# make_plot(epochs, 'Loss', f'myFilename.png {learning_rate}', train_loss, val_loss)
#%%

def calculate_best_score(scoring, metric, reverse=False):
    resulting_dict = {}
    for name, performance_metrics in scoring.items():
        resulting_dict[name] = {
            'train_loss': min(performance_metrics['train_loss']), 
            'train_acc': max(performance_metrics['train_acc']), 
            'val_loss': min(performance_metrics['val_loss']), 
            'val_acc': max(performance_metrics['val_acc']),
            'ece': performance_metrics['ece'],
            #'val_ece': performance_metrics['val_ece']
        }
    
    return sorted(resulting_dict.items(), key=lambda x: x[1][metric], reverse=reverse)

#%% Hyperparameter search

learning_rates = [0.001, 0.01]
hidden_nodes = [16,64, 256]
memory_sizes = [4,8,16,32]

scoring = {}

# Options
# - 'Kalman'
# - 'LSTM'
# - 'FF'
model_type = 'Kalman'

for learning_rate in learning_rates:
    for n_hidden_nodes in hidden_nodes:
        for memory_size in memory_sizes:
            name = f'{model_type}_LR-{learning_rate}_Hnodes-{n_hidden_nodes}_mem-{memory_size}'
            log.info(f'Starting {name}')
            model, optimizer = get_model(model_type=model_type, 
                                         learning_rate=learning_rate, 
                                         hidden_nodes=n_hidden_nodes, 
                                         layers=None)
            
            train_loss, train_acc, val_loss, val_acc = run_epochs(n_epochs=EPOCH, 
                                                                 model=model, 
                                                                 optimizer=optimizer, 
                                                                 memory_size=memory_size,
                                                                 data=test, 
                                                                 validation=validation)
            #global final_model = model
            
            scoring[name] = {
                'train_loss': train_loss, 
                'train_acc': train_acc, 
                'val_loss': val_loss, 
                'val_acc': val_acc
            }
            
#import pickle
calculate_best_score(scoring, 'val_acc', reverse=True)
calculate_best_score(scoring, 'train_acc', reverse=True)

#with open(f'{model_type}_score.pkl', 'wb') as f:
 #   pickle.dump(scoring, f)
    

#%% Final training 

learning_rate = 0.001 #0.01
hidden_nodes = 4 #10 bedst, prov 96
memory_size = 16 #32 # med 10 epoch
model_type = 'LSTM'

model, optimizer = get_model(model_type=model_type, 
                                 learning_rate=learning_rate, 
                                 hidden_nodes= hidden_nodes, 
                                 layers=None)

train_loss, train_acc, val_loss, val_acc, ece = run_epochs(n_epochs=EPOCH, 
                                                         model=model, 
                                                         optimizer=optimizer, 
                                                         memory_size=memory_size,
                                                         train_data=train, 
                                                         val_data=test)


name = f'{model_type}_LR-{learning_rate}_Hnodes-{hidden_nodes}_mem-{memory_size}'
scoring = {}
scoring[name] = {
    'train_loss': train_loss, 
    'train_acc': train_acc, 
    'val_loss': val_loss, 
    'val_acc': val_acc,
    'ece': ece
}


#%% Create plots

for key, val in scoring.items():
    train_loss = val['train_loss']
    train_acc = val['train_acc']
    val_loss = val['val_loss']
    val_acc = val['val_acc']
    
    make_plot(EPOCH, 'Loss', name + "_loss.png", ('Training', train_loss), ('Validation', val_loss))
    make_plot(EPOCH, 'Acc', name + "_acc.png", ('Training', train_acc), ('Validation', val_acc))
    
    make_plot(EPOCH, 'Loss', name + "_loss.png", ('Training', train_loss), ('Validation', val_loss), save=True)
    make_plot(EPOCH, 'Acc', name + "_acc.png", ('Training', train_acc), ('Validation', val_acc), save=True)



#%% Save model for implementation

if SAVE_MODEL == True:
    torch.save(model, f'Final_acro_{model_type}_E_{EPOCH}_lr_{learning_rate}_hn_{hidden_nodes}_ms{memory_size}.pt')
    