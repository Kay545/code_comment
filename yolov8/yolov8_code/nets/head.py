import torch 
import math 
import torch.nn as nn        
from nets.layers import *

class Head(nn.Module):
    def __init__(self,in_channels,cls_num=6,reg_max=16,width_ratio=0.25):
        super(Head,self).__init__()
        Creg = max(16,math.ceil(256*width_ratio)//4,4*reg_max)
        in_channels = math.ceil(in_channels)
        self.bbox_branch = nn.Sequential(
            CBL(in_channels,Creg,3),
            CBL(Creg,Creg,3),
            Conv(Creg,4*reg_max,1)
        )


        Ncls = max(math.ceil(256*width_ratio),cls_num)
        self.cls_branch = nn.Sequential(
            CBL(in_channels,Ncls,3),
            CBL(Ncls,Ncls,3),
            Conv(Ncls,cls_num,1)
        )

    def forward(self,x):
        reg_out = self.bbox_branch(x)
        cls_out = self.cls_branch(x)
        out = torch.cat([reg_out,cls_out],dim=1)
        return out 


        