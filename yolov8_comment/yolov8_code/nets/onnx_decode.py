import torch 
import torch.nn as nn                   

class Onnx_Decode(nn.Module):
    def __init__(self,input_size,class_num=2):
        super(Onnx_Decode,self).__init__()
        self.input_w = input_size[0]
        self.input_h = input_size[1]
        self.reg_max = 16
        self.proj = torch.arange(self.reg_max)
        self.use_dfl = True  
        self.class_num = class_num
        
    def pred_process(self,outputs):
        '''
            L = class_num + 4*self.reg_max = class_num + 64
            多尺度结果bxLx80x80,bxLx40x40,bxLx20x20,整合到一起为 b x 8400 x L 
            按照cls 与 box 拆分为 b x 8400 x 2 , b x 8400 x 64
        '''
        predictions = [] # 记录每个尺度的转换结果 
        strides = [] # 记录每个尺度的缩放倍数
        for output in outputs:
            self.bs,cs,in_h,in_w = output.shape 
            # 计算该尺度特征图相对于网络输入的缩放倍数
            stride = self.input_h //in_h
            strides.append(stride)
            # shape 转换 如b x 66 x 80 x 80 -> b x 66 x 6400 -> b x 6400 x 66
            prediction = output.view(self.bs,4*self.reg_max+self.class_num,-1).permute(0,2,1).contiguous()
            predictions.append(prediction)
        # b x (6400+1600+400)x 66= b x 8400 x 66
        predictions = torch.cat(predictions,dim=1)
        # 按照cls 与 reg 进行拆分
        # 类别用sigmoid方法，对每个类别进行二分类
        pred_scores = predictions[...,4*self.reg_max:]
        pred_regs = predictions[...,:4*self.reg_max]
        return pred_scores,pred_regs,strides
    
    def decode(self,pred_regs):
        '''
            预测结果解码
            1. 对bbox预测回归的分布进行积分
            2. 结合anc_points，得到所有8400个像素点的预测结果
        '''
        if self.use_dfl:
            b,a,c = pred_regs.shape # b x 8400 x 64 
            # 分布通过 softmax 进行离散化处理
            pred_regs = pred_regs.view(b,a,4,c//4).softmax(3)
            # 积分，相当于对16个分布值进行加权求和
            pred_regs = pred_regs.matmul(self.proj.type(self.FloatTensor))

        # 此时的regs,shape-> bx8400x4,其中4表示 anc_point中心点分别距离预测box的左上边与右下边的距离
        lt = pred_regs[...,:2]
        rb = pred_regs[...,2:]
        # xmin ymin 
        x1y1 = (self.anc_points - lt)
        # xmax ymax
        x2y2 = (self.anc_points + rb )
        # b x 8400 x 4        
        pred_bboxes = torch.cat([x1y1,x2y2],dim=-1)
        return pred_bboxes
    
    def make_anchors(self,strides,grid_cell_offset=0.5):
        '''
            各特征图每个像素点一个锚点即Anchors,即每个像素点只预测一个box
            故共有 80x80 + 40x40 + 20x20 = 8400个anchors
        '''
        # anc_points : 8400 x 2 ，每个像素中心点坐标
        # strides_tensor: 8400 x 1 ，每个像素的缩放倍数
        anc_points,strides_tensor = [],[]
        for i , stride in enumerate(strides):
            in_h = self.input_h//stride 
            in_w = self.input_w//stride 
            
            # 
            sx = torch.arange(0,in_w,).type(self.FloatTensor) + grid_cell_offset
            sy = torch.arange(0,in_h).type(self.FloatTensor) + grid_cell_offset
            # in_h x in_w
            grid_y,grid_x = torch.meshgrid(sy,sx)
            # in_h x in_w x 2 -> N x 2
            anc_points.append(torch.stack((grid_x,grid_y),-1).view(-1,2).type(self.FloatTensor))
            strides_tensor.append(torch.full((in_h*in_w,1),stride).type(self.FloatTensor))
        
        return torch.cat(anc_points,dim=0),torch.cat(strides_tensor,dim=0)
    
    def forward(self,outputs):
        self.cuda = True  if outputs[0].is_cuda else False 
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor 
        
        # 预测结果预处理
        pred_scores,pred_regs,strides = self.pred_process(outputs)
        self.anc_points,self.stride_scales = self.make_anchors(strides)
        pred_bboxes = self.decode(pred_regs)
        
        result = torch.cat([pred_bboxes.detach()*self.stride_scales,pred_scores.detach().sigmoid()],dim=-1)
        return result 
        
        