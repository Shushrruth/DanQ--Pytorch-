
'''

DanQ
Model architecture


'''


import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F
from torch.nn import MaxPool1d

class Net(nn.Module):
    def __init__(self, in_channels=4, out_channels=320):
        super(Net, self).__init__()
        self.Conv = nn.Conv1d(in_channels = 4, out_channels = 320, kernel_size = 26)
        
        self.Maxpool = nn.MaxPool1d(kernel_size = 13, stride = 13)
        
        self.Drop = nn.Dropout(0.1)
        
        self.BiLSTM = nn.LSTM(input_size = 320, hidden_size = 320, num_layers=2,
                                 batch_first = True,
                                 dropout = 0.5,
                                 bidirectional = True)
    
        self.Linear1 = nn.Linear(75*320*2, 925)
        
        self.Linear2 = nn.Linear(925, 1)
        
    def forward(self, input):
        x = self.Conv(input)
        
        x = F.relu(x)
        
        x = self.Maxpool(x)
        
        x = self.Drop(x)
        
        x = torch.transpose(x, 1, 2)
        
        x,_ = self.BiLSTM(x)
        
        x = torch.flatten(x,1)
        
        x = self.Linear1(x)
        
        x = F.relu(x)
        
        x = self.Linear2(x)
        
        x = torch.sigmoid(x)

        return x 

k=torch.rand(1,4,1000).cuda()
Net().cuda()(k.float())

