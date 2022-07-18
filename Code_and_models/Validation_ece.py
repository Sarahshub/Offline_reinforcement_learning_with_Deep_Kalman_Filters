# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:30:02 2022

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
from torch.utils.data import DataLoader
from kalman_filter import Kalman_filter
from LSTM_Model import RNNNetwork
from ff_baseline import Net

n_choices = 3
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

#%% Load data

if use_cuda:
    env = torch.load('training_data_400_q_learning_acro.pt')
else:
    env = torch.load('training_data_400_q_learning_acro.pt', map_location=torch.device('cpu'))


#%% unpack data

'''data = []
for k1, v1 in env.items():
    value = [v2 for k2, v2 in v1.items()]
    data.append(value)'''
    

data = []
for k,v in env.items():
    if isinstance(v, dict):
        data.append(list(v.values()))
    elif isinstance(v, list):
        data.append(list(v))
    else:
        raise Exception(f'Unsupported loading of data, 2nd layer type is: {type(v)}')
    
#%%
val = data
   


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
            raise Exception(f'No group: "{group}" found in performance logs', group)
        return dictionary[group]
    
    def add_predict_target(self, pred, target, group):
        self._add_to_list_in_dict(self.target, group, target)
        self._add_to_list_in_dict(self.predictions, group, pred)
#%% helper methods
# Creates an input tensor of enum actions into binned classes of 0 and 1
# i.e. input of [1,0,0,1] becomes [[0,1], [1,0], [1,0], [0,1]]
# i.e. input of [1,0,0,1] becomes [[0,1], [1,0], [1,0], [0,1]]
def classify_action(action, n_choices=n_choices):
    return F.one_hot(action, n_choices)
    

#%%
def validation(observations, monitor, epoch, model):

     model.eval()
     with torch.no_grad():
         batch_state, batch_action, batch_next_state, batch_reward = zip(*observations)
         

         state = torch.stack(batch_state)
         last_action = batch_action[-1]
         action = last_action.long() if torch.is_tensor(last_action) else LongTensor([last_action])
         #action = batch_action.view(-1).float() # experiment with feeding actions, but not current action becuase of prediction. 
         #reward = batch_reward.view(-1,1) not relevant for now or maybe ever
         #next_state = batch_next_state.view(-1, 4) not relevant for now or maybe ever
         if len(state.shape) > 2:
             state = state.squeeze(0)
             
         result = model(state) # feed past 10-20 timesteps and rest as is
         if len(result.shape) > 1:
             result = result.squeeze()
         #loss_fn = torch.nn.MSELoss()
         loss_fn = torch.nn.CrossEntropyLoss()
         
         classifier = classify_action(action).float()
         
         loss = loss_fn(result.view(1,-1), classifier.view(1,-1))
         monitor.log_loss(group=epoch, loss=loss.item())
     
     
         #Calculate accuracy
           
         models_action = result.argmax(dim = 0)
         accuracy = models_action.eq(action.flatten())
         monitor.log_accuracy(group=epoch, accuracy=accuracy.item())     
         
         # save data for ECE 
         monitor.add_predict_target(group=epoch, target=action.item(), pred=result)
#%%

#EPOCH = 10 # number of episodes
#LR = 0.001  # NN optimizer learning ratee
#WEGHT_DECAY = 0.000
LOAD_MODEL = False # Load model to get expert data
SAVE_MODEL = False # Don't overwrite exsisting model  

#INPUT_SIZE = 4
#hidden_nodes = 32
#layer_dimension = 2
#output_dimension = 2
#MEMORY_SIZE = 16

#%%


#model = torch.load('BCmodel_Val_acc95.4.pt', map_location=torch.device('cpu'))

# LSTM Model
#model = torch.load("LSTM_test.pt")

# Kalman Filter
#model = torch.load("kalman_model.pt")
model = torch.load("Final_acro_3l_Kalman_E_3_lr_0.001_hn_64_ms32.pt")
# Feed forward network
#model = torch.load("Baseline_FF.pt")

model.eval()
 

#%% Load parameters
#model_type = 'Kalman' # choose between FF for feed forward network, LSTM for RNN LSTM, Kalman for the kalman filter
EPOCH = 1 # number of episodes
#LR = 0.001  # NN optimizer learning ratee
#WEGHT_DECAY = 0.000
LOAD_MODEL = False # Load model to get expert data
SAVE_MODEL = False # Don't overwrite exsisting model  
INPUT_SIZE = 6
#hidden_nodes = 32
layer_dimension = 2
output_dimension = 3
MEMORY_SIZE = 1


#%%
def run_epoch(epoch, monitor, model, dataset, func, memory_size):
    total_episodes = len(dataset)
    current_episode = 0
    for episode in dataset:
        current_episode = current_episode + 1
        #if current_episode == 0:
            #print(f"Epoch: {epoch:03} Running episode: {current_episode:03}/{total_episodes:03}")
        #else:
            #print(f"\rEpoch: {epoch:03} Running episode: {current_episode:03}/{total_episodes:03}")
            
        #model.clear_memory()
        rounds = len(episode) - memory_size
        if rounds <= 0: 
            continue
        
        for i in range(rounds):
            func(episode[i:i + memory_size], monitor, epoch, model)

#%%
def run_epochs(n_epochs, model, validation, memory_size=None):
    
    val_loss = []
    val_acc = []
    
    validation_monitor = PerformanceMonitor()
   
    for epoch in range(n_epochs):
        log.info('===============================')
        log.info(f'Running Epoch {epoch}')
        
        
        log.info('== Running validation for Epoch ==')
        run_epoch(epoch, validation_monitor, model, val, validation, memory_size = MEMORY_SIZE)
        
        
        e_loss_val = Tensor(validation_monitor.get_losses(group=epoch)).mean().item()
        log.info(f'validation loss per epoch: {e_loss_val}')
        val_loss.append(e_loss_val)
        
        
        e_acc = Tensor(validation_monitor.get_accuracies(group=epoch)).mean().item()
        log.info(f'Validation acc per epoch: {e_acc}')
        val_acc.append(e_acc)
        
        
    
    
    log.info(f'All {EPOCH} epochs complete')
    
    return (val_loss, val_acc, validation_monitor)
#%%

def calculate_best_score(scoring, metric, reverse=False):
    resulting_dict = {}
    for name, performance_metrics in scoring.items():
        resulting_dict[name] = {
            'val_loss': min(performance_metrics['val_loss']), 
            'val_acc': max(performance_metrics['val_acc']),
            'ece': performance_metrics['ece'],
            #'val_ece': performance_metrics['val_ece']
        }
    
    return sorted(resulting_dict.items(), key=lambda x: x[1][metric], reverse=reverse)
#%% Run 

val_loss, val_acc, ece = run_epochs(n_epochs=EPOCH, model=model, validation=validation)


name = "LSTM"
scoring = {}
scoring[name] = {
    'val_loss': val_loss, 
    'val_acc': val_acc,
    'ece': ece
}




#%%
targets = []
for k,v in ece.target.items():
    targets = targets + v
    
predictions = []
for k,v in ece.predictions.items():
    predictions = predictions + v


#%%
import torch.nn.functional as F



def _calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

#%%
a = torch.stack(predictions)
a_soft = F.softmax(a, dim = 1)
b = Tensor(targets)

bins = 10

e=_calculate_ece(a_soft,b, n_bins=bins)


#%% Plot ECE 
import numpy as np

def make_model_diagrams(outputs, labels, n_bins=2):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    #print(softmaxes, confidences, predictions, accuracies, overall_accuracy)
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    #print(bin_indices)
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
   
    
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = _calculate_ece(outputs, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    #plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('reliability_diagram.png')
    plt.show()
    return ece
#%%

make_model_diagrams(a.cpu(), b.cpu(), n_bins=bins)

print(e)
#%%
#%% mean and standard deviation LSTM
import statistics as s
# for memory size 16 af 10 gange
a = [0.9477, 0.9476, 0.9477, 0.9477, 0.9477, 0.9476, 0.9477, 0.9477, 0.9477, 0.9476]
e = [0.3916, 0.3915, 0.3916, 0.3916, 0.3916, 0.3915, 0.3916, 0.3916, 0.3916, 0.3915]
ee = [0.0021, 0.0020, 0.0021, 0.0022, 0.0020, 0.0020, 0.0021, 0.0022,0.0021, 0.0022]
a_stv = s.stdev(a)
a_mu = s.mean(a)
print(a_stv, a_mu) 
e_stv = s.stdev(e)
e_mu = s.mean(e)        
print(e_stv, e_mu) 

ee_stv = s.stdev(ee)
ee_mu = s.mean(ee)       
print(ee_stv, ee_mu)   
#%% Kalman
# for memory size 16 af 10 gange
a = [0.8850, 0.8826, 0.8831, 0.8824, 0.8820, 0.8828, 0.8824, 0.8826, 0.8823, 0.8828,]
e = [0.3687, 0.3501, 0.3506, 0.3500, 0.3491, 0.3502, 0.3500, 0.3501, 0.3499, 0.3502]
ee = [0.0384, 0.0384, 0.0379, 0.0370, 0.0369, 0.0378, 0.0368, 0.0365, 0.0347, 0.0378]
a_stv = s.stdev(a)
a_mu = s.mean(a)
print(a_stv, a_mu) 
e_stv = s.stdev(e)
e_mu = s.mean(e)        
print(e_stv, e_mu)  

ee_stv = s.stdev(ee)
ee_mu = s.mean(ee)       
print(ee_stv, ee_mu)

#for i in range(9):
#%% NN 
a = [0.9441, 0.9442, 0.9442, 0.9442, 0.9442, 0.9442, 0.9442, 0.9441, 0.9442, 0.9442]
e = [0.3970, 0.3970, 0.3970, 0.3970, 0.3970, 0.3970, 0.3970, 0.3970, 0.3970, 0.3970]
ee = [0.0237, 0.0237, 0.0237, 0.0237, 0.0237, 0.0237, 0.0237, 0.0237, 0.0237, 0.0237]
a_stv = s.stdev(a)
a_mu = s.mean(a)
print(a_stv, a_mu) 
e_stv = s.stdev(e)
e_mu = s.mean(e)        
print(e_stv, e_mu)  

ee_stv = s.stdev(ee)
ee_mu = s.mean(ee)       
print(ee_stv, ee_mu)
    