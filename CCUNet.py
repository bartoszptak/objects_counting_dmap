#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import segmentation_models_pytorch as smp


class Block(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,  padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,  padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class CCUNet(torch.nn.Module):
    def __init__(self, encoder_name):
        super(CCUNet, self).__init__()
        self.input1 = torch.nn.Conv2d(3, 64, kernel_size=1)
        self.input2 = torch.nn.Conv2d(2, 64, kernel_size=1)
        
        self.rgb_block = Block(inplanes=64, planes=64, stride=1)
        self.dis_block = Block(inplanes=64, planes=64, stride=1)
        
        self.network = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=128, classes=1)


    def forward(self, x_rgb, x_dis):
        
        features_rgb = self.rgb_block(self.input1(x_rgb))        
        features_dis = self.dis_block(self.input2(x_dis))
        
        concat = torch.cat((features_rgb, features_dis), 1)
        out = self.network(concat)
        
        return out

    def predict(self, x_rgb, x_dis):

        if self.training:
            self.eval()

        with torch.no_grad():
            x = self(x_rgb, x_dis)

        return x