from calendar import c
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
from models.LLAMA2 import TransformerVAEwithTaskID, ModelArgsVAE, ModelArgs 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
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
class NCFDataset2(Dataset):
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


class NCFDataset(Dataset):
    def __init__(self):
        train_path = "data/llama.train.doubleside.npz"
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
                #datas.append(d)

                info = readdisp(os.path.join(basedir, fn))

                disps.append(info["disp1"]) 
                disps.append(info["disp1"]) 
                #disps.append(info["disp1"]) 

                masks.append(info["mask1"])
                masks.append(info["mask1"])
                #masks.append(info["mask1"])
                
                p1, p2 = info["loc1"], info["loc2"] 
                d = degrees2kilometers(locations2degrees(p1[1], p1[0], p2[1], p2[0]))
                
                dists.append(d) 
                dists.append(d) 
                #dists.append(d) 
                
                infos.append(0)
                infos.append(1)
                #infos.append(2)
            
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
        else:
            b = x2[:1024]
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
    xs = torch.stack(xs, dim=1) 
    ds = torch.vstack(ds) 
    ks = torch.tensor(ks, dtype=torch.float32)
    ms = torch.vstack(ms)
    return xs, ds, ks, ms
import torch.nn as nn

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

from models.disp4 import DispModel
def main(args):
    # 训练有一个不是X1
    version = "v6.1"# 全新训练
    model_name = f"ckpt_large/llama.{version}.pt" #保存和加载神经网络模型权重的文件路径
    #cp ckpt_large/llama.v1.2.pt ckpt_large/llama.v1.3.pt
    train_dataset = NCFDataset()     
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch, num_workers=3)
    device = torch.device("cuda:0")
    model_args = ModelArgs() 
    model = TransformerVAEwithTaskID(model_args)
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("no model")
    model.to(device)
    model.train()
    #for var in model.parameters():
    #    if len(var.shape)<2:continue
    #    nn.init.xavier_uniform_(var, gain=nn.init.calculate_gain('relu '))
    lossfn = WMSELoss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间r
    isfixed = False    
    if isfixed:
        for key, var in model.named_parameters():
            if var.dtype != torch.float32:continue # BN统计计数无梯度
            if "decoder_disp" in key: # 仅有最后一层有out
                var.requires_grad = True
            else:
                var.requires_grad = False  
    model.to(device)
    model.train()
    #lossfn = Loss() 
    #lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/llama.{version}.txt", "a") #记录训练过程中的loss

    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-3)
    n_epoch = 100   
    count = 0 
    for b in range(n_epoch):
        for x, d, k, m in train_dataloader:
            #print(x.shape, d.shape, m.shape)
            x = x.to(device)
            x = x.permute(1, 0).unsqueeze(2) 
            #x = torch.cat([x, x, x], dim=2)
            #s = torch.zeros([x.shape[0], 48, x.shape[2]], device=device)
            #x = torch.cat([x, s], dim=1)
            #print(x.shape)
            m = m.to(device) 
            d = d.to(device)
            y1, y2, mu, logvar = model(x)
            #print(mu, logvar)

            #print(y1.shape, y2.shape)
            #print(y1.shape, y2.shape, hidden.shape)
            #lc = -1-logvar + mu ** 2 + logvar.exp() 
            #print(lc)
            nllloss = torch.sum(
            -1-logvar + mu ** 2 + logvar.exp(), dim = 2
            ).sum()

            loss1, loss2, loss3, loss = lossfn(y1, y2, d, m) 
            loss = loss + nllloss 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            count += 1        
            if count % 50 ==1:
                loss = loss.detach().cpu().numpy() 
                loss1 = loss1.detach().cpu().numpy() 
                loss2 = loss2.detach().cpu().numpy()
                loss3 = loss3.detach().cpu().numpy()
                nllloss = nllloss.detach().cpu().numpy() 
                outloss.write(f"{b},{count},{loss},{loss1},{loss2},{loss3},{nllloss}\n")
                print(b, count, loss, loss1, loss2, loss3)
                p1 = y1.detach().cpu().numpy()[0] 
                p2 = y2.detach().cpu().numpy()[0]
                #x = x.detach().cpu().numpy()[0]
                d = d.cpu().numpy()[0]
                torch.save(model.state_dict(), model_name)  
                t = np.linspace(3, 51, 48)    
                m = m.cpu().numpy()[0]    
                fig = plt.figure(1, figsize=(12, 6), dpi=300) 
                gs = gridspec.GridSpec(1, 2)
                ax = fig.add_subplot(gs[0, 0])
                w = x[0]
                t2 = np.arange(len(w))*0.1 
                #ax.plot(t2, w, lw=1, c="b")
                ax.plot(t, p1, lw=1, c="r") 
                ax.plot(t, d, c="b") 
                ax.scatter(t[m!=0], d[m!=0], c="b")
                ax.scatter(t[p2>0.5], p1[p2>0.5], c="r")     
                error = [[] for i in range(48)]           
                with torch.no_grad():
                    for i in range(50):
                        y1, y2, mu, logvar = model(x[:2])
                        #logit1, logit2 = output[:, -48:, 0], output[:, -48:, 1]
                        #y1 = logit1#.sigmoid() * 6 
                        #y2 = logit2.sigmoid()
                        p1 = y1.cpu().numpy()[0]
                        for k in range(48): 
                            error[k].append(p1[k])
                        ax.plot(t, p1, c="g", lw=0.5, alpha=0.1)
                
                ax.set_ylim(2.5, 4)
                ax.set_xlim(0, 53)
                

                ax = fig.add_subplot(gs[0, 1]) 
                ax.violinplot(error,
                  showmeans=True)
                ax.set_xticks([y*5 + 1 for y in range(len(error)//5)],
                                labels=[f'{y*5+3}' for y in range(len(error)//5)])
                ax.scatter(t[m!=0]-2, d[m!=0], c="b", label="Manual")
                ax.legend()
                ax.set_xlabel('Period [s]')
                ax.set_ylabel('Velocity [km/s]')
                ax.set_title('Violin plot of Dispersion')
                plt.savefig(f"tempfig/llama.{version}.png")
                plt.cla() 
                plt.clf()
                plt.close()
        outloss.write(f"#{b},{count},{loss}\n")  
#nohup python llama.train.v1.py > logdir/llama.v1.3.log 2>&1 &
#971631
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                         
    args = parser.parse_args()
    main(args)




