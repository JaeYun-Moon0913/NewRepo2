from collections.abc import Sequence
from torch import outer
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2) #(55-3) /2 +1  = 27
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2) # 13

        )
        # self.conv2 = nn.Conv2d(96,256,5,1,2,padding_mode='zeros') # (27-5+4)/1 + 1 = 27 
        self.conv3 = nn.Conv2d(256,384,3,1,1,padding_mode='zeros') # 13 
        self.conv4 = nn.Conv2d(384,384,3,1,1,padding_mode='zeros') # 13 
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,256,3,1,1,padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        #self.conv5 = nn.Conv2d(384,256,3,1,1,padding_mode='zeros')  # 13 

        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        
        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))
        out = F.dropout(out,0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out,0.5)
        out = self.fc3(out)
        out = F.log_softmax(out,dim=1)

        return out 

        