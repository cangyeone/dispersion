from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import numpy as np 
import re 
import os 
import obspy 
from obspy.geodetics import degrees2kilometers, locations2degrees 
import tqdm 
import scipy.interpolate as interpolate 
from scipy.signal import resample
import obspy.signal.interpolation as obspy_interpolate
def sline(line):
    return [i for i in line.split(" ") if len(i)>0]
def readdisp(path):
    file_ = open(path, "r", encoding="utf-8") 
    temp = sline(file_.readline()) 
    loc1 = [float(temp[0]), float(temp[1])] 
    temp = sline(file_.readline()) 
    loc2 = [float(temp[0]), float(temp[1])]  
    disp1 = [] 
    #disp2 = []
    freq1 = []
    mask1 = []
    for line in file_.readlines():
        temp = sline(line.replace("NaN", "0.0000")) 
        v = float(temp[1]) 
        if np.abs(v)>1e-3:
            disp1.append(v)
            mask1.append(1.0) 
        else:
            disp1.append(0)
            mask1.append(0.0)
        #disp2.append(float(temp[2]))
    file_.close()
    return {"loc1":loc1, "loc2":loc2, "disp1":np.array(disp1), "mask1":np.array(mask1)}
class NCFDataset(Dataset):
    def __init__(self):
        train_path = "data/llama.train.npz"
        if os.path.exists(train_path):
            file_ = np.load(train_path) 
            self.datas = file_["datas"] 
            self.disps = file_["disps"] 
            self.dists = file_["dists"] 
            self.masks = file_["masks"] 
            self.infos = file_["infos"] 
            self.length = len(self.datas)
        else:
            basedir = "data/NCF/X1_ALL_ZZ" 
            file_names = os.listdir(basedir) 
            ncfdict = {}
            for fn in file_names:
                nsplit = fn.split(".") 
                if nsplit[0] != "X1" or nsplit[2] != "X1":
                    path = os.path.join(basedir, fn) 
                    ncfdict[fn] = path 
            basedir = "data/NCF/FilterDis" 
            file_names = os.listdir(basedir) 
            datas = []
            disps = []
            dists = []
            masks = []
            infos = []
            for fn in tqdm.tqdm(file_names):
                fkey = fn.replace("CDisp.T.", "") 
                if fkey not in ncfdict:continue
                path = ncfdict[fkey] 
                d = obspy.read(path)[0].data 
                d -= np.mean(d) 
                d /= (np.std(d)+1e-6) 

                datas.append(d)
                datas.append(d)
                datas.append(d)

                info = readdisp(os.path.join(basedir, fn))

                disps.append(info["disp1"]) 
                disps.append(info["disp1"]) 
                disps.append(info["disp1"]) 

                masks.append(info["mask1"])
                masks.append(info["mask1"])
                masks.append(info["mask1"])
                p1, p2 = info["loc1"], info["loc2"] 
                d = degrees2kilometers(locations2degrees(p1[1], p1[0], p2[1], p2[0]))
                dists.append(d) 
                dists.append(d) 
                dists.append(d) 
                infos.append(0)
                infos.append(1)
                infos.append(2)
            np.savez(train_path, datas=datas, disps=disps, dists=dists, masks=masks, infos=infos) 
            self.dists = np.array(dists) 
            self.datas = np.array(datas)
            self.disps = np.array(disps) 
            self.masks = np.array(masks)
            self.infos = np.array(infos)
            self.length = len(self.datas)
            print(self.datas.shape, self.infos.shape)
    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        #idx = idxc % self.length 
        x = self.datas[idx] 
        #x = self.datas[idx] 
        i = self.infos[idx]
        #print(x.shape)
        #b = x[3600-512:3600-512+1024]
        x1 = x[:3600][::-1].copy()
        x2 = x[3601:]
        if i == 0:
            b = x1[:1024]
        elif i == 1:
            b = x2[:1024]
        else:
            b = x1[:1024] + x2[:1024]
        #b = resample(b, 10240)
        x = torch.tensor(b, dtype=torch.float32)
        x -= x.mean() 
        #x /= (x.abs().max()+1e-6)
        x /= x.std() + 1e-6 
        #x += np.random.normal(0, 1.0, size=x.shape) * np.random.uniform(0.2, 5.0)
        d = self.disps[idx]
        d = torch.tensor(d, dtype=torch.float32)
        k = self.dists[idx] 
        m = torch.tensor(self.masks[idx], dtype=torch.float32)
        return (x, d, k, m)

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds, ks, ms = [], [], [], []
    for x, d, k, m in batch:
        xs.append(x) 
        ds.append(d)
        ks.append(k)
        ms.append(m)
    xs = torch.stack(xs, dim=0) 
    ds = torch.vstack(ds) 
    ks = torch.tensor(ks, dtype=torch.float32)
    ms = torch.vstack(ms)
    return xs, ds, ks, ms
