import torch 
import torch.nn as nn              
from utils.utils import *
from loss.self_assigner import TaskAlignedAssiger
from loss.box_loss import BboxLoss
class YOLOv8_Loss(nn.Module):
    def __init__(self,class_num,input_size):
        super(YOLOv8_Loss,self).__init__()
        self.class_num = class_num
        self.input_w = input_size[0]
        self.input_h = input_size[1]
        self.reg_max = 16
        self.use_dfl = True 
        self.proj = torch.arange(self.reg_max)
        self.assigner = TaskAlignedAssiger(topk=10, num_classes=class_num, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)  
        self.box_weights = 7.5
        self.cls_weights = 0.5
        self.dfl_weights = 1.5
    def pred_process(self,inputs):
        '''
            L = class_num + 4*self.reg_max = class_num + 64
            多尺度结果bxLx80x80,bxLx40x40,bxLx20x20,整合到一起为 b x 8400 x L 
            按照cls 与 box 拆分为 b x 8400 x 2 , b x 8400 x 64
        '''
        predictions = [] # 记录每个尺度的转换结果 
        strides = [] # 记录每个尺度的缩放倍数
        for input in inputs:
            self.bs,cs,in_h,in_w = input.shape 
            # 计算该尺度特征图相对于网络输入的缩放倍数
            stride = self.input_h // in_h 
            strides.append(stride)
            # shape 转换 如 b x 80 x 80 x cls_num+2 -> b x 6400 x cls_num+2
            prediction = input.view(self.bs,4*self.reg_max+self.class_num,-1).permute(0,2,1).contiguous()
            predictions.append(prediction)
        # b x (6400+1600+400)x cls_num+2 = b x 8400 x (cls_num + 2)
        predictions = torch.cat(predictions,dim=1)
        # 按照cls 与 reg 进行拆分
        # 类别用sigmoid方法，对每个类别进行二分类
        pred_scores = predictions[...,4*self.reg_max:]
        pred_regs = predictions[...,:4*self.reg_max]
        return pred_scores,pred_regs,strides 

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
        x1y1 = self.anc_points - lt 
        # xmax ymax
        x2y2 = self.anc_points + rb 
        # b x 8400 x 4        
        pred_bboxes = torch.cat([x1y1,x2y2],dim=-1)
        return pred_bboxes
    
    def ann_process(self,annotations):
        '''
            batch内不同图像标注box个数可能不同，故进行对齐处理
            1. 按照batch内的最大box数目M,新建全0tensor
            2. 然后将实际标注数据填充与前面，如后面为0，则说明不足M，用0补齐
        '''
        # 获取batch内每张图像标注box的bacth_idx
        batch_idx = annotations[:,0]
        # 计算每张图像中标注框的个数
        # 原理对tensor内相同值进行汇总
        _,counts = batch_idx.unique(return_counts=True)
        counts = counts.type(torch.int32)
        # 按照batch内最大M个GT创新全0的tensor b x M x 5 ,其中5 = cx,cy,width,height，cls
        res = torch.zeros(self.bs,counts.max(),5).type(self.FloatTensor)
        for j in range(self.bs):
            matches = batch_idx == j 
            n = matches.sum()
            if n: 
                res[j,:n] = annotations[matches,1:]
        # res 为归一化之后的结果,需通过scales映射回输入尺度
        scales = [self.input_w,self.input_h,self.input_w,self.input_h]
        scales = torch.tensor(scales).type(self.FloatTensor)
        res[...,:4] = xywh2xyxy(res[...,:4]).mul_(scales)
        # gt_bboxes b x M x 4
        # gt_labels b x M x 1
        gt_bboxes,gt_labels = res[...,:4],res[...,4:]
        # gt_mask b x M 
        # 通过对四个坐标值相加，如果为0，则说明该gt信息为填充信息，在mask中为False，
        # 后期计算过程中会进行过滤
        gt_mask = gt_bboxes.sum(2,keepdim=True).gt_(0)
        return gt_bboxes,gt_labels,gt_mask

        
        
    def forward(self,inputs,annotations):
        self.cuda = True  if inputs[0].is_cuda else False 
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor 
        
        # ---------- 预测结果预处理 ---------- #
        # 将多尺度输出整合为一个Tensor,便于整体进展矩阵运算
        pred_scores,pred_regs,strides = self.pred_process(inputs)
        
        # --------- 生成anchors锚点 ---------#
        # 各尺度特征图每个位置一个锚点Anchors(与yolov5中的anchors不同,此处不是先验框)
        # 表示每个像素点只有一个预测结果
        self.anc_points,self.stride_scales = self.make_anchors(strides)
        
        # -------------     解码 ------------- #
        # 预测回归结果解码到bbox xmin,ymin,xmax,ymax格式
        pred_bboxes = self.decode(pred_regs)
        
        # ---------- 标注数据预处理 ----------- #
        gt_bboxes,gt_labels,gt_mask = self.ann_process(annotations)
        
        # ----------- 正负样本筛选 ------------ #
        target_bboxes,target_scores,fg_mask= self.assigner(pred_scores.detach().sigmoid(),
                                                        pred_bboxes.detach()*self.stride_scales,
                                                        self.anc_points*self.stride_scales,
                                                        gt_labels,
                                                        gt_bboxes,
                                                        gt_mask)
        # 正样本个数
        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3,device=inputs[0].device)
        # cls_loss 
        loss[1] = self.bce(pred_scores,target_scores).sum()/target_scores_sum
        
        # bbox loss
        if fg_mask.sum():
            target_bboxes /= self.stride_scales
            loss[0],loss[2] = self.bbox_loss(pred_regs,pred_bboxes,self.anc_points,target_bboxes,target_scores,
                                             target_scores_sum,fg_mask)
            
        loss[0] *= self.box_weights # box gain
        loss[1] *= self.cls_weights # cls gain
        loss[2] *= self.dfl_weights  # dfl gain
        
        return loss.sum(),loss.detach()