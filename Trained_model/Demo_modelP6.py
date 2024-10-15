import torch
import numpy as np
import scipy.io as scio

io = r'./data.mat'
l1 = scio.loadmat(io) 
init_input = torch.Tensor(l1['data']).transpose(0,1) 
input_feature = init_input.shape[1]
model_path = r'./Best_model_p1_lat.pth'

class MNN(torch.nn.Module):
    def __init__(self):
        super(MNN,self).__init__()    
        self.all = torch.nn.Sequential(
            torch.nn.Linear(input_feature,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,1024),
            torch.nn.ReLU(),    
            torch.nn.Linear(1024,256),
            torch.nn.ReLU(),                     
            torch.nn.Linear(256,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,3),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        output2 = self.all(x)
        return output2

MN = MNN()

statedict = torch.load(model_path, map_location='cuda:0')
MN.load_state_dict(statedict)
MN.eval()

if isinstance(init_input, torch.Tensor):
    output = MN(init_input)
    doutput = output.cpu().detach().numpy()
    scio.savemat(r'./modelout.mat', {'output': doutput})
else:
    raise TypeError("Expected init_input to be a Tensor")
