#coding:utf-8
import torch
import math 
import torch.nn as nn



# --------------------- #
#   Conv 输出层的卷积
# --------------------- #
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,bias=False):
        super(Conv,self).__init__()
        padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias)
    def forward(self,x):
        x = self.conv(x)
        return x
# ---------------------- #
#   CBL 基础卷积块
# Conv + BN  + leaky_relu
# ---------------------- #
class CBL(nn.Module):
    def __init__(self,in_channels,
                 out_channels=128,
                 kernel_size=3,
                 stride=1,
                 use_silu=False,
                 dynamic = False,
                 max_channels=512,
                 width_ratio=0.25):
        super(CBL,self).__init__()
        if dynamic:
            out_channels = math.ceil(max_channels*width_ratio)
        in_channels = math.ceil(in_channels)
        self.conv = Conv(in_channels,out_channels,kernel_size,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU() if use_silu else nn.LeakyReLU(0.1)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x 

# ------------------------ #
#       SiLU激活函数
# ------------------------ #
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU,self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = x * self.sigmoid(x)
        return out  

# ------------------------------------- #
#   Dosample = Maxpool + Conv(k=3,s=2)
# ------------------------------------- #
class DownSample(nn.Module):
    def __init__(self,in_channels,max_channels=512,width_ratio=0.25):
        super(DownSample,self).__init__()
        in_channels = math.ceil(in_channels)
        out_channels = math.ceil(max_channels*width_ratio)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.cbl1 = CBL(in_channels,out_channels//2,1)
        
        self.cbl2 = CBL(in_channels,out_channels//2,3,2)
        
    def forward(self,x):
        x1 = self.maxpool(x)
        x1 = self.cbl1(x1)
        x2 = self.cbl2(x)
        out = torch.cat([x1,x2],dim=1)
        return out 
    
# ------------------------- #
#           SPPF
# ------------------------- #
class SPPF(nn.Module):
    def __init__(self,in_channels,pool_size=5):
        super(SPPF,self).__init__()
        in_channels = math.ceil(in_channels)
        self.cbl_1 = CBL(in_channels,in_channels,1)
        self.cbl_2 = CBL(in_channels*4,in_channels,1)
        
        self.maxpool1 = nn.MaxPool2d(pool_size,1,pool_size//2)
        self.maxpool2 = nn.MaxPool2d(pool_size,1,pool_size//2)
        self.maxpool2 = nn.MaxPool2d(pool_size,1,pool_size//2)
    def forward(self,x):
        x = self.cbl_1(x)
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x1)
        x3 = self.maxpool2(x2)
        out = torch.cat([x,x1,x2,x3],dim=1)
        out = self.cbl_2(out)
        return out 

# ----------------------------- #
#       Upsample = Dconv
# ----------------------------- #
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSample,self).__init__()
        in_channels = math.ceil(in_channels)
        out_channels = math.ceil(out_channels)
        self.cbl = CBL(in_channels,out_channels,1)
        self.upsample = nn.ConvTranspose2d(
                        out_channels,
                        out_channels,
                        2,
                        2,
                        bias=True
                        )
        
    def forward(self,x):
        x = self.cbl(x)
        out = self.upsample(x)
        return out 
    


# ------------------------------- #
#     Bottleneck1 有残差边
# ------------------------------- #
class Bottleneck_block(nn.Module):
    def __init__(self,in_channels,out_channels,with_res=True):
        super(Bottleneck_block,self).__init__()
        self.with_res = with_res
        self.cbl1 = CBL(in_channels,in_channels//2,3)
        self.cbl2 = CBL(in_channels//2,out_channels,3)
        
    def forward(self,x):
        x1 = self.cbl1(x)
        x2 = self.cbl2(x1)
        if self.with_res:
            x2 = x + x2  
        return x2 
# ------------------------------- #
#             C2f1_X
# ------------------------------- #
    
class C2f1_X(nn.Module):
    def __init__(self,in_channels,max_channels=512,max_res_num=3,depth_ratio=0.33,width_ratio=0.25):
        super(C2f1_X,self).__init__()
        in_channels = math.ceil(in_channels)
        self.split_channels = in_channels//2
        out_channels = math.ceil(max_channels * width_ratio)
        self.cbl1 = CBL(in_channels,out_channels,1)
        self.res_num = math.ceil(max_res_num*depth_ratio)
        self.res = nn.ModuleList([Bottleneck_block(out_channels//2,out_channels//2) for i in range(self.res_num)])
        self.cbl2 = CBL(out_channels//2*(self.res_num+2),out_channels,1)
        
    def forward(self,x):
        x = self.cbl1(x)
        x1,x2 = x.split((self.split_channels,self.split_channels),dim=1)
        fs = [x1,x2] 
        for i in range(self.res_num):
            x3 = self.res[i](fs[-1])
            fs.append(x3)
        out = self.cbl2(torch.cat(fs,dim=1))
        return out 

# ------------------------------- #
#             C2f1_X
# ------------------------------- #
    
class C2f2_X(nn.Module):
    def __init__(self,in_channels,max_channels=512,max_res_num=3,depth_ratio=0.33,width_ratio=0.25):
        super(C2f2_X,self).__init__()
        
        out_channels = math.ceil(max_channels * width_ratio)
        self.split_channels = out_channels//2
        self.cbl1 = CBL(in_channels,out_channels,1)
        self.res_num = math.ceil(max_res_num*depth_ratio)
        self.res = nn.ModuleList([Bottleneck_block(out_channels//2,out_channels//2,with_res=False) for i in range(self.res_num)])
        self.cbl2 = CBL(out_channels//2*(self.res_num+2),out_channels,1)
        
    def forward(self,x):
        x = self.cbl1(x)
        x1,x2 = x.split((self.split_channels,self.split_channels),dim=1)
        fs = [x1,x2] 
        for i in range(self.res_num):
            x3 = self.res[i](fs[-1])
            fs.append(x3)
        out = self.cbl2(torch.cat(fs,dim=1))
        return out 

