import torch 
import torch.nn as nn             
from nets.layers import *
from nets.backbone import Backbone 
from nets.neck import Neck
from nets.head import Head
from nets.onnx_decode import Onnx_Decode 
class YOLOv8(nn.Module):
    def __init__(self,net_version='yolov8n',class_num=2,reg_max=16,input_size=None,onnx_export=False):
        super(YOLOv8,self).__init__()
        self.onnx_export = onnx_export
        net_ratio_dict ={'yolov8n':[0.33,0.25],'yolov8s':[0.33,0.5]}
        depth_ratio,width_ratio = net_ratio_dict[net_version]
        self.backbone = Backbone(net_ratio_dict,net_version=net_version)
        
        self.neck = Neck(net_ratio_dict,net_version=net_version)
        
        self.head_8  = Head(256*width_ratio,cls_num=class_num,reg_max=reg_max,width_ratio=width_ratio)
        self.head_16 = Head(512*width_ratio,cls_num=class_num,reg_max=reg_max,width_ratio=width_ratio)
        self.head_32 = Head(1024*width_ratio,cls_num=class_num,reg_max=reg_max,width_ratio=width_ratio)
        
        
        if self.onnx_export:
            self.decode = Onnx_Decode(input_size,class_num=class_num)
        
        
    def forward(self,x):
        outputs = self.backbone(x)
        out8,out16,out32 = self.neck(outputs)
        pred_8 = self.head_8(out8)
        pred_16 = self.head_16(out16)
        pred_32 = self.head_32(out32)
        
        if self.onnx_export:
            preds = self.decode([pred_8,pred_16,pred_32])
            return preds 
        else:
            return [pred_8,pred_16,pred_32]
        