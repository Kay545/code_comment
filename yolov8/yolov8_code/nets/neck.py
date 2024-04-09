import torch 
import torch.nn as nn               
from nets.layers import *

class Neck(nn.Module):
    def __init__(self,net_ratio_dict,net_version='yolov8n'):
        super(Neck,self).__init__()
        
        depth_ratio,width_ratio = net_ratio_dict[net_version]
        self.upsample_32 = UpSample(1024*width_ratio,
                                    1024*width_ratio)
        self.c2f2_x_1 = C2f2_X(1536*width_ratio,
                               max_channels=512,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        
        self.upsample_16 = UpSample(512*width_ratio,
                                    512*width_ratio)
        self.c2f2_x_2 = C2f2_X(768*width_ratio,
                               max_channels=256,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        
        self.cbl_1 = CBL(256*width_ratio,
                         kernel_size=3,
                        stride=2,
                        dynamic=True,
                        max_channels=256,
                        width_ratio=width_ratio)
        self.c2f2_x_3 = C2f2_X(768*width_ratio,
                               max_channels=512,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        

        self.cbl_2 = CBL(512*width_ratio,
                         kernel_size=3,
                        stride=2,
                        dynamic=True,
                        max_channels=512,
                        width_ratio=width_ratio)
        self.c2f2_x_4 = C2f2_X(1536*width_ratio,
                               max_channels=1024,
                               max_res_num=3,
                               depth_ratio=depth_ratio,
                               width_ratio=width_ratio)
        
        
    def forward(self,inputs):
        out8,out16,out32 = inputs 
        
        x1 = self.upsample_32(out32)
        x1 = torch.cat([x1,out16],dim=1)
        x1 = self.c2f2_x_1(x1)
        
        x2 = self.upsample_16(x1)
        x2 = torch.cat([x2,out8],dim=1)
        x2 = self.c2f2_x_2(x2)
        
        x3 = self.cbl_1(x2)
        x3 = torch.cat([x3,x1],dim=1)
        x3 = self.c2f2_x_3(x3)
        
        x4 = self.cbl_2(x3)
        x4 = torch.cat([x4,out32],dim=1)
        x4 = self.c2f2_x_4(x4)
        
        return [x2,x3,x4]