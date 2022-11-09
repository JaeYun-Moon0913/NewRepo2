import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

class VGG(nn.Module):
    def __init__(self, input_channel, label_num, model_name, input_size = 224, batchnorm = False):
        super(VGG,self).__init__()
        self.vgg11 = {1:[64],2:[128],3:[256,256],4:[512,512],5:[512,512]}
        self.vgg13 = {1:[64,64],2:[128,128],3:[256,256],4:[512,512],5:[512,512]}
        self.vgg16 = {1:[64,64],2:[128,128],3:[256,256,256],4:[512,512,512],5:[512,512,512]}
        self.vgg19 = {1:[64,64],2:[128,128],3:[256,256,256,256],4:[512,512,512,512],5:[512,512,512,512]}
        

        if model_name == 'vgg11':
            model_name = self.vgg11
        elif model_name == 'vgg13':
            model_name = self.vgg13
        elif model_name == 'vgg16':
            model_name = self.vgg16
        elif model_name == 'vgg19':
            model_name = self.vgg19

        else : 
            raise Exception("input vgg model name error")

        self.conv1 = make_conv(model_name[1],3,batchnorm)
        # self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = make_conv(model_name[2],64,batchnorm)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = make_conv(model_name[3],128,batchnorm)
        # self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = make_conv(model_name[4],256,batchnorm)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = make_conv(model_name[5],512,batchnorm)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer = make_fc(input_size,label_num)

        # self.fc_layer = nn.Sequential(
        #     # CIFAR10은 크기가 32x32이므로 
        #     nn.Linear(512*7*7, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 1000),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1000, label_num),
        # )
        
    def forward(self,x):
        x = self.conv1(x)
        # out = self.Maxpool1(x)
        x = self.conv2(x)
        # out = self.Maxpool2(x)
        x = self.conv3(x)
        # out = self.Maxpool3(x)
        x = self.conv4(x)
        # out = self.Maxpool4(x)
        x = self.conv5(x)
        # out = self.Maxpool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x



def make_conv(list_conv,input_channels,batchnorm):
    layers = []
    in_planes = input_channels
    for value in list_conv:
        if batchnorm == False: 
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_planes = value
        else : 
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d())# 논문에는 없지만 개선을 위해 추가 
            layers.append(nn.ReLU())
            in_planes = value
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def make_fc(input_size,label_num):
    layers = []

    if input_size == 224: 
        layers.append(nn.Linear(512*7*7, 4096))
    elif input_size == 32: 
        layers.append(nn.Linear(512*1*1, 4096))
    else :
        pass 
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(4096,4096))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(4096,1000))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(1000, label_num))

    return nn.Sequential(*layers)

