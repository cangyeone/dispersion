from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150


import math 
class DistEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("rge", torch.arange(192).unsqueeze(1))
    def forward(self, x, dist):
        T, B, C = x.shape 
        embx = self.rge * dist.unsqueeze(0)
        x = x + torch.sin(embx.unsqueeze(2)/1000) 
        return x  

class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, stride, ks):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nout, ks, stride=stride, padding=(ks-1)//2), 
            nn.BatchNorm1d(nout), 
            nn.LeakyReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class ResBlock(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(nin, nin//2, 1, 1), 
            ConvBNReLU(nin//2, nin, 1, 5), 
        )
    def forward(self, x):
        y = self.layers(x) 
        return y + x 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = 32
        self.layers1 = nn.Sequential(
            ConvBNReLU(1, base*1, 1, 5), 
            nn.Conv1d(base*1, base*2, 5, 2, padding=2), 
            ResBlock(base*2), 
            nn.Conv1d(base*2, base*3, 5, 2, padding=2), 
            ResBlock(base*3), 
            nn.Conv1d(base*3, base*4, 5, 2, padding=2), 
            ResBlock(base*4), 
            nn.Conv1d(base*4, base*5, 5, 2, padding=2), 
            ResBlock(base*5), 
            nn.Conv1d(base*5, base*6, 5, 2, padding=2), 
            ResBlock(base*6), 
            nn.Conv1d(base*6, base*7, 5, 2, padding=2), 
            ResBlock(base*7), 
            nn.Conv1d(base*7, base*8, 5, 2, padding=2), 
            ResBlock(base*8), 
            nn.Conv1d(base*8, base*9, 5, 2, padding=2), 
            ResBlock(base*9), 
            nn.Conv1d(base*9, base*10, 5, 2, padding=2), 
            ResBlock(base*10), 
            nn.Conv1d(base*10, base*11, 5, 2, padding=2), 
            ResBlock(base*11), 
        )
        self.layers2 = nn.Sequential(
            nn.Linear(100, base*11)
        )
        self.logvar = nn.parameter.Parameter(torch.zeros([1, 100]))
        self.mu = nn.parameter.Parameter(torch.zeros([1, 100]))
        self.out1 = nn.Linear(base*11*2, 48)
        self.out2 = nn.Linear(base*11*2, 48)
    def forward(self, x):
        x = x.unsqueeze(1) 
        T, B, C = x.shape 
        h1 = self.layers1(x) 
        h1 = h1.squeeze()

        std = torch.exp(self.logvar * 0.5) 
        eps = torch.randn([x.shape[0], 100], device=x.device, dtype=x.dtype)

        s = self.mu + eps * std

        h2 = self.layers2(s)
        h = torch.cat([h1, h2], dim=1)

        y1 = self.out1(h) 
        y2 = self.out2(h)
        y1 = torch.sigmoid(y1) * 6 
        y2 = torch.sigmoid(y2)
        return y1, y2, self.mu, self.logvar 
    
class ModelInfer(nn.Module):
    def __init__(self):
        super().__init__()
        base = 32
        self.layers1 = nn.Sequential(
            ConvBNReLU(1, base*1, 1, 5), 
            nn.Conv1d(base*1, base*2, 5, 2, padding=2), 
            ResBlock(base*2), 
            nn.Conv1d(base*2, base*3, 5, 2, padding=2), 
            ResBlock(base*3), 
            nn.Conv1d(base*3, base*4, 5, 2, padding=2), 
            ResBlock(base*4), 
            nn.Conv1d(base*4, base*5, 5, 2, padding=2), 
            ResBlock(base*5), 
            nn.Conv1d(base*5, base*6, 5, 2, padding=2), 
            ResBlock(base*6), 
            nn.Conv1d(base*6, base*7, 5, 2, padding=2), 
            ResBlock(base*7), 
            nn.Conv1d(base*7, base*8, 5, 2, padding=2), 
            ResBlock(base*8), 
            nn.Conv1d(base*8, base*9, 5, 2, padding=2), 
            ResBlock(base*9), 
            nn.Conv1d(base*9, base*10, 5, 2, padding=2), 
            ResBlock(base*10), 
            nn.Conv1d(base*10, base*11, 5, 2, padding=2), 
            ResBlock(base*11), 
        )
        self.layers2 = nn.Sequential(
            nn.Linear(100, base*11)
        )
        self.logvar = nn.parameter.Parameter(torch.zeros([1, 100]))
        self.mu = nn.parameter.Parameter(torch.zeros([1, 100]))
        self.out1 = nn.Linear(base*11*2, 48)
        self.out2 = nn.Linear(base*11*2, 48)
    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(1)
        x -= x.mean() 
        x /= x.std() + 1e-6 
        
        h1 = self.layers1(x.clone()) 
        h1 = h1.squeeze(2)

        std = torch.exp(self.logvar * 0.5) 
        eps = torch.randn([x.shape[0], 100], device=x.device, dtype=x.dtype)

        s = self.mu + eps * std

        h2 = self.layers2(s)
        h = torch.cat([h1, h2], dim=1)

        y1 = self.out1(h) 
        y2 = self.out2(h)
        y1 = torch.sigmoid(y1) * 6 
        y2 = torch.sigmoid(y2)
        return y1, y2

class WMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y1, y2, d, m):
        loss1 = torch.square(y1-d)*m 
        loss1 = loss1.sum() 
        loss2 = (- m * torch.log(y2+1e-6) - (1-m) * torch.log(1-y2+1e-6)).sum()
        loss3 = ((y1[:, 1:] - y1[:, :-1])**2).mean()
        loss2 = loss2 * 0.001
        loss3 = loss3 * 0.00
        return loss1, loss2, loss3, loss1 + loss2 + loss3