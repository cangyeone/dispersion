import numpy as np 
import torch 


method = torch.jit.load("ckpt/disp.cnn.jit")
method.eval()
K = 50 # use 50 sampled data to estimate the uncertainty. 
y1s, y2s = [], []
with torch.no_grad():
    x = torch.randn([1024])
    y1, y2 = method(x)
    for k in range(K):
        x = torch.randn([1024])
        y1, y2 = method(x)
        y1s.append(y1)
        y2s.append(y2)
    y1s = torch.concat(y1s, dim=0)
    y2s = torch.concat(y2s, dim=0)
    y1s = y1s.cpu().numpy()
    y2s = y2s.cpu().numpy()
ofile = open("logdir/disp.cnn.txt", "w")
y1 = np.mean(y1s, axis=0)
y2 = np.mean(y2s, axis=0) 
e1 = np.std(y1s, axis=0)
e2 = np.std(y2s, axis=0)
ofile.write("velocity,filtered,standerd deviation\n")
for a, b, c in zip(y1, y2, e1):
    ofile.write("{:.4f},{},{:.4f}\n".format(a, b>0.5, c))
