from models.cnn import ModelInfer as Model1
from models.trans import TransformerVAEwithTaskIDInfer as Model2# can be used as Transformer model 
import torch 

model = Model1()
model.eval()
model.load_state_dict(torch.load("ckpt/cnn.pt", map_location="cpu", weights_only=True))
model.to("cpu") 

torch.jit.save(torch.jit.script(model), "ckpt/disp.cnn.jit")