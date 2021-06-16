# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import *

class ESANet(nn.Module):
    def __init__(self, num_classes=37):
        super(ESANet, self).__init__()
        
        # RGB Encoder
        self.rgb_conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )
        self.fusion1 = RGBD_Fusion(64)
        self.rgb_block1 = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                ResNet_Layer1(),
                )
        self.fusion2 = RGBD_Fusion(64)
        self.rgb_resnet2 = ResNet_Layer2()
        self.fusion3 = RGBD_Fusion(128)
        self.rgb_resnet3 = ResNet_Layer3()
        self.fusion4 = RGBD_Fusion(256)
        self.rgb_resnet4 = ResNet_Layer4()
        self.fusion5 = RGBD_Fusion(512)
        self.context_module = Context_Module(2, 3, 4)
        
        # Depth Encoder
        self.depth_conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )
        self.depth_block1 = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                ResNet_Layer1(),
                )
        self.depth_resnet2 = ResNet_Layer2()
        self.depth_resnet3 = ResNet_Layer3()
        self.depth_resnet4 = ResNet_Layer4()
        
        # Decoder
        self.decoder1 = Decoder_Module(512, 512)
        self.decoder2 = Decoder_Module(512, 256)
        self.decoder3 = Decoder_Module(256, 128)
        self.decoder_block1 = nn.Sequential(
                nn.Conv2d(128, num_classes, 3, padding=1),
                Learned_Upx2(num_classes),
                Learned_Upx2(num_classes),
                )
        
        # Decoder_shortcut
        self.decoder_shortcut512 = nn.Sequential(
                nn.Conv2d(256, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                )
        self.decoder_shortcut256 = nn.Sequential(
                nn.Conv2d(128, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                )
        self.decoder_shortcut128 = nn.Sequential(
                nn.Conv2d(64, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                )
        
    def forward(self, depth, rgb):
        # 第一次融合
        depth_out = self.depth_conv1(depth)
        rgb_out = self.rgb_conv1(rgb)
        rgb_out = self.fusion1(depth_out, rgb_out)
        rgb_out = self.rgb_block1(rgb_out)
        
        # 第二次融合（生成短连接128）
        depth_out = self.depth_block1(depth_out)
        rgb_shortcut128 = self.fusion2(depth_out, rgb_out)        
        
        # 第三次融合（生成短连接256）
        depth_out = self.depth_resnet2(depth_out)
        rgb_out = self.rgb_resnet2(rgb_shortcut128)
        rgb_shortcut256 = self.fusion3(depth_out, rgb_out)
        
        # 第四次融合（生成短连接512）
        depth_out = self.depth_resnet3(depth_out)
        rgb_out = self.rgb_resnet3(rgb_shortcut256)
        rgb_shortcut512 = self.fusion4(depth_out, rgb_out)
        
        # 第五次融合
        depth_out = self.depth_resnet4(depth_out)
        rgb_out = self.rgb_resnet4(rgb_shortcut512)
        out = self.fusion5(depth_out, rgb_out)
        out = self.context_module(out)
        
        # 第一次解码
        shortcut1 = self.decoder_shortcut512(rgb_shortcut512)
        out = self.decoder1(out, shortcut1)
        
        # 第二次解码
        shortcut2 = self.decoder_shortcut256(rgb_shortcut256)
        out = self.decoder2(out, shortcut2)
        
        # 第三次解码
        shortcut3 = self.decoder_shortcut128(rgb_shortcut128)
        out = self.decoder3(out, shortcut3)
        
        # 分割结果
        out = self.decoder_block1(out)
        
        return out
    
if __name__ == '__main__':
    net = ESANet().to('cuda:0')
    depth = torch.randn(2, 1, 480, 640).to('cuda:0')
    rgb = torch.randn(2, 3, 480, 640).to('cuda:0')
    
    out = net(depth, rgb)