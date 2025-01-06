import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from models_torch import *
from utils import *
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.zeros_(m.bias.data)
    elif isinstance(m, nn.GRU):
        for value in m.state_dict():
            if 'weight_ih' in value:
                #print(value,param.shape,'Orthogonal')
                init.xavier_normal_(m.state_dict()[value])
            elif 'weight_hh' in value:
                init.orthogonal_(m.state_dict()[value])
            elif 'bias' in value:
                init.zeros_(m.state_dict()[value])
    elif isinstance(m, nn.GRUCell):
        for value in m.state_dict():
            if 'weight_ih' in value:
                #print(value,param.shape,'Orthogonal')
                init.xavier_normal_(m.state_dict()[value])
            elif 'weight_hh' in value:
                init.orthogonal_(m.state_dict()[value])
            elif 'bias' in value:
                init.zeros_(m.state_dict()[value])

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def train(epoch, reps=20):
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, sample in enumerate(train_loader):
        
        data, target = sample['x'].to(device), sample['y'].to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, target)
        loss.backward()
        train_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        """if batch_idx % 1000 == 0:
            print("{} secs".format(time.time() - start_time))
            print('====> Epoch: {} Batch {}/{} Average loss: {:.4f}'.format(
            epoch, batch_idx+1, len(Y)//batch_size, train_loss / (batch_idx+1)), end='\r', flush=True)
            start_time = time.time()"""

    print('====> Epoch: {} Average loss: {:.10f}'.format(
        epoch, train_loss / (batch_idx+1)), flush=True)
    return train_loss / (batch_idx+1)

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--k', type=int, default='4',
                        help='The value of k')
    parser.add_argument('--w', type=int, default='3',
                        help='The value of w')
    parser.add_argument('--file_name', type=str, default='xor10_small',
                        help='The name of the input file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Name for the log file')
    parser.add_argument('--epochs', type=int, default='12',
                        help='Num of epochs')
    parser.add_argument('--model_weights_path', type=str, default='file_bstrap',
                        help='Path to model parameters')
    parser.add_argument('--timesteps', type=int, default='64',
                        help='Num of time steps')
    return parser

t1 = time.time()

parser = get_argument_parser()
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
num_epochs=FLAGS.epochs

batch_size=5120
timesteps=FLAGS.timesteps
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using", device)
sequence = np.load(FLAGS.file_name  + "_" + str(FLAGS.k) + "_" + str(FLAGS.w) + ".npy")

vocab_size = pow(4, FLAGS.k)#len(np.unique(sequence))
sequence = sequence.astype(np.int64)

# Convert data into context and target symbols
X, Y = generate_single_output_data(sequence, batch_size, timesteps)

kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
train_dataset = CustomDL(X, Y)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **kwargs)

# Model_Y
dic = {'vocab_size': vocab_size, 'emb_size': 16,
        'length': timesteps, 'jump': 16,
        'hdim1': 128, 'hdim2': 128, 'n_layers': 1,
        'bidirectional': True
       }

print("Vocab Size {}".format(vocab_size))

# Create Bootstrap Model
model = BootstrapNN(**dic).to(device)
# Apply Weight Initalization
model.apply(weight_init)
# Learning Rate Decay
mul = len(Y)/5e7
decayrate = mul/(len(Y) // batch_size)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-3)
fcn = lambda step: 1./(1. + decayrate*step)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fcn)

# Training with Best Model Selection
epoch_loss = 1e8
for epoch in range(num_epochs):
    lss = train(epoch+1)
    if lss < epoch_loss:
        torch.save(model.state_dict(), FLAGS.model_weights_path)
        print("Loss went from {:.4f} to {:.4f}".format(epoch_loss, lss))
        epoch_loss = lss



model_size = os.stat(FLAGS.model_weights_path).st_size
print('Model size: {} B'.format(model_size))
print("Done")
t2 = time.time()
print('Training model time consume: {}'.format(t2-t1))
print('Training Peak GPU memory usage: {} KBs'.format(torch.cuda.max_memory_allocated() // 1024))



