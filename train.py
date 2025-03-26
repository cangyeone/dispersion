from utils.data import NCFDataset, collate_batch
from torch.utils.data import DataLoader
from models.cnn import Model as Model1, WMSELoss
from models.trans import TransformerVAEwithTaskID as Model2 
import torch 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
def main(args):
    model_name = args.checkpoint #保存和加载神经网络模型权重的文件路径
    train_dataset = NCFDataset()     
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch, num_workers=3)
    device = torch.device("cuda")
    if args.model == "cnn":
        model = Model1()
    elif args.model == "trans":
        model = Model2()
    else:
        print("model not found")
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("model not found")
    model.to(device)
    model.train()
    lossfn = WMSELoss() 
    lossfn.to(device)

    outloss = open(f"logdir/{model_name}.loss.txt", "a") #记录训练过程中的loss

    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-3)
    n_epoch = 100   
    count = 0 
    for b in range(n_epoch):
        for x, d, k, m in train_dataloader:
            x = x.to(device)
            x = x.permute(1, 0)
            m = m.to(device) 
            d = d.to(device)
            y1, y2, mu, logvar = model(x)

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
                plt.savefig(f"logdir/train.{model_name}.png")
                plt.cla() 
                plt.clf()
                plt.close()
        outloss.write(f"#{b},{count},{loss}\n")  
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")              
    parser.add_argument('-m', '--model', default="cnn", help="model name")                       
    parser.add_argument('-c', '--checkpoint', default="ckpt/cnn.pt", help="checkpoint name")                                  
    args = parser.parse_args()
    main(args)




