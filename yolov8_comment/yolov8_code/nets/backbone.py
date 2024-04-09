import torch 
import torch.nn as nn     
from nets.layers import *            

class Backbone(nn.Module):
    def __init__(self,net_ratio_dict,net_version='yolov8n'):
        super(Backbone,self).__init__()
        depth_ratio ,width_ratio = net_ratio_dict[net_version]
        self.cbl = CBL(3,
                       kernel_size=3,
                       stride=2,
                       dynamic=True,
                       max_channels=64,
                       width_ratio=width_ratio)
        

        self.downsample_1 = DownSample(64*width_ratio,
                                       max_channels=128,
                                       width_ratio=width_ratio)
        self.c2f1_x_1 = C2f1_X(128*width_ratio,
                               max_channels=128,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        
        
        self.downsample_2 = DownSample(128*width_ratio,
                                       max_channels=256,
                                       width_ratio=width_ratio)
        self.c2f1_x_2 = C2f1_X(256*width_ratio,
                               max_channels=256,
                               max_res_num=6,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        

        self.downsample_3 = DownSample(256*width_ratio,
                                       max_channels=512,
                                       width_ratio=width_ratio)
        self.c2f1_x_3 = C2f1_X(512*width_ratio,
                               max_channels=512,
                               max_res_num=6,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)

        self.downsample_4 = DownSample(512*width_ratio,
                                       max_channels=1024,
                                       width_ratio=width_ratio)
        self.c2f1_x_4 = C2f1_X(1024*width_ratio,
                               max_channels=1024,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        
        self.sppf = SPPF(1024*width_ratio)
        
    def forward(self,x):
        x = self.cbl(x)
        
        x = self.downsample_1(x)
        x = self.c2f1_x_1(x)
        
        x = self.downsample_2(x)
        out8 = self.c2f1_x_2(x)
        
        x1 = self.downsample_3(out8)
        out16 = self.c2f1_x_3(x1)
        
        x2 = self.downsample_4(out16)
        x2 = self.c2f1_x_4(x2)
        out32 = self.sppf(x2)
        
        return [out8,out16,out32]
    