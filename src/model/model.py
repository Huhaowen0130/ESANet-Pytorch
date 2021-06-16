# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBD_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(RGBD_Fusion, self).__init__()
        
        self.conv_layers_rgb = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid(), 
                 )
        
        self.conv_layers_depth = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid(), 
                 )
        
    def forward(self, depth, rgb):        
        depth_1 = F.adaptive_avg_pool2d(depth, (1, 1))
        depth_1 = self.conv_layers_depth(depth_1)
        depth_1 = torch.mul(depth, depth_1)
        
        rgb_1 = F.adaptive_avg_pool2d(rgb, (1, 1))
        rgb_1 = self.conv_layers_rgb(rgb_1)
        rgb_1 = torch.mul(rgb, rgb_1)
        
        out = depth_1 + rgb_1
        
        return out
    
    
class Context_Module(nn.Module):
    def __init__(self, b, pbh, pbw):
        super(Context_Module, self).__init__()
        
        self.sub_branches = []
        
        for _ in range(b - 1):
            self.sub_branches.append(nn.Sequential(
                    nn.AvgPool2d((pbh, pbw), stride=(pbh, pbw)),
                    nn.Conv2d(512, 512 // b, 1),
                    nn.BatchNorm2d(512 // b),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=(3, 4)),
                    ))
        self.sub_branches = nn.ModuleList(self.sub_branches)
            
        self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(512, 512 // b, 1),
                nn.BatchNorm2d(512 // b),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=(15, 20)),
                )
        
        self.final_layers = nn.Sequential(
                nn.Conv2d(512 * 2, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                )
        
    def forward(self, x):
        out = x
        
        for f in self.sub_branches:
            out = torch.cat((out, f(x)), -3)
            
        out = torch.cat((out, self.global_branch(x)), -3)
        
        out = self.final_layers(out)
        
        return out
    
    
class Learned_Upx2(nn.Module):
    def __init__(self, in_channels):
        super(Learned_Upx2, self).__init__()
        
        self.layers = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                )
        
    def forward(self, x):
        out = self.layers(x)
        
        return out
    
    
class NBt1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NBt1D, self).__init__()
        
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(out_channels),
                )
        
    def forward(self, x):
        out = self.layers(x)
        out += x
        out = F.relu(out)
        
        return out
    
    
class Decoder_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Module, self).__init__()
        
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                NBt1D(out_channels, out_channels),
                NBt1D(out_channels, out_channels),
                NBt1D(out_channels, out_channels),
                Learned_Upx2(out_channels),
                )
        
    def forward(self, x, shortcut):
        out = self.layers(x)
        out += shortcut
        
        return out
    
    
class ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride=1):
        super(ResNet_Block, self).__init__()
        
        self.first_stride = first_stride
        
        if self.first_stride == 1:
            self.main_branch = NBt1D(in_channels, out_channels)
        else:
            self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 1), stride=(2, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(out_channels),
                )
            self.res_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 1), stride=(2, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(out_channels),
                )
            
    def forward(self, x):
        out = self.main_branch(x)
        
        if self.first_stride == 1:
            out += x
        else:
            out += self.res_branch(x)
            
        out = F.relu(out)
        
        return out
    
    
class ResNet_Layer1(nn.Module):
    def __init__(self):
        super(ResNet_Layer1, self).__init__()
        
        self.layers = nn.Sequential(
                ResNet_Block(64, 64),
                ResNet_Block(64, 64),
                ResNet_Block(64, 64),
                )
    def forward(self, x):
        out = self.layers(x)
        
        return out
    
    
class ResNet_Layer2(nn.Module):
    def __init__(self):
        super(ResNet_Layer2, self).__init__()
        
        self.layers = nn.Sequential(
                ResNet_Block(64, 128, 2),
                ResNet_Block(128, 128),
                ResNet_Block(128, 128),
                ResNet_Block(128, 128),
                )
    def forward(self, x):
        out = self.layers(x)
        
        return out
    
    
class ResNet_Layer3(nn.Module):
    def __init__(self):
        super(ResNet_Layer3, self).__init__()
        
        self.layers = nn.Sequential(
                ResNet_Block(128, 256, 2),
                ResNet_Block(256, 256),
                ResNet_Block(256, 256),
                ResNet_Block(256, 256),
                ResNet_Block(256, 256),
                ResNet_Block(256, 256),
                )
    def forward(self, x):
        out = self.layers(x)
        
        return out
    
    
class ResNet_Layer4(nn.Module):
    def __init__(self):
        super(ResNet_Layer4, self).__init__()
        
        self.layers = nn.Sequential(
                ResNet_Block(256, 512, 2),
                ResNet_Block(512, 512),
                ResNet_Block(512, 512),
                )
    def forward(self, x):
        out = self.layers(x)
        
        return out