import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
#import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
#import xlrd
#import math
import scipy.io as scio
import time
import os
from datetime import datetime, timedelta  
today = (datetime.now()).strftime('%m%d')  

nowloc = os.path.split(os.path.realpath(__file__))[0]

io1 = r'dataset\P1dataset.mat'
datapath = os.path.join(nowloc,io1)
model_save_path_best = os.path.join(nowloc,'/Best_model_p1.pth')
l1 = scio.loadmat(datapath) 

testcase = 0
init_input = torch.Tensor(l1['data']).transpose(0,1) 
target = torch.Tensor(l1['label']).transpose(0,1)


Total_BS = target.shape[0]
# hyper parameters
EPOCH = 10000
LR = 1e-4
BATCH_SIZE = 16
num_of_test = round(Total_BS*0.15)

fin_step = round(init_input.shape[0]/BATCH_SIZE)
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    init_input = init_input.to(device)
    target  = target .to(device)
    


input_feature = init_input.shape[1]

train_dataset = Data.TensorDataset(init_input, target)

train, test, valid = Data.random_split(dataset= train_dataset, lengths=[init_input.shape[0]-2*num_of_test,num_of_test,num_of_test],generator = torch.Generator().manual_seed(54))

train_loader = Data.DataLoader(dataset=train, batch_size= BATCH_SIZE, shuffle= True)
test_loader = Data.DataLoader(dataset= test, batch_size = num_of_test,shuffle= False)
val_loader = Data.DataLoader(dataset= valid, batch_size = num_of_test,shuffle= False)

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
if torch.cuda.is_available():
    MN = MN.to(device)

optimizer2 = torch.optim.Adam(MN.parameters(), lr = LR,weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,mode='min', factor = 0.9,verbose = True, patience = 50000, cooldown = 1,eps = 1e-8)
loss_func = torch.nn.MSELoss()
val_loss_best = 0.02

start = time.time()
if testcase == 0:
    for epoch in range(EPOCH):
        total_loss = 0
        #testloss = 0
        for step, (b_x,b_y) in enumerate(train_loader):
            MN.train()
            output = MN(b_x)
            loss = loss_func(output,b_y)     
            optimizer2.zero_grad()  
            loss.backward()    
            optimizer2.step()
            scheduler2.step(loss)
            
            total_loss += loss 
        if  (epoch+1) % 10 == 0:
            with torch.no_grad():
                for x,y in val_loader:
                    MN.eval()
                    val_output = MN(x)
                    testloss = loss_func(val_output,y)
                    print('EPOCH:',epoch+1)
                    print('MSELOSS:',loss.data.item())
                    print('TESTloss:',testloss.data.item())
                
                if testloss < val_loss_best:
                    val_loss_best = testloss
                    for x,y in test_loader:
                        MN.eval()
                        val_output = MN(x)
                        testloss = loss_func(val_output,y)
                    vout = val_output.cpu().detach().numpy()
                    vlabel = y.cpu().detach().numpy()
                    torch.save(MN.state_dict(), model_save_path_best)
                    sig = x.cpu().detach().numpy()
                    scio.savemat('/home/zwangfd/new/pogo/pogo/mp2/ML/python/Xianqu_project/dataset/testbest_P1.mat',{'valout' : vout, 'y' : vlabel,'sig':sig})

    print(val_loss_best)
    endtime = time.time()
    print(endtime-start)

else: 
    statedict = torch.load(model_save_path_best)
    cnn.load_state_dict(statedict)    
    cnn.eval()
    Posout = cnn(testinput)
    Posout = Posout.cpu().detach().numpy()
    testlabel = testlabel.cpu().detach().numpy()
    scio.savemat('/home/zwangfd/new/pogo/pogo/mp2/ML/python/impactTest/Dataset/LandE_testout_tc_'+str(today)+'.mat',{'testlabel':testlabel,'Posout':Posout,'Enlabel':enlabel,'minE':minE,'maxE':maxE})
    print('Test complete')
