import torch 
import torch.nn as nn 
from utils.utils import *

class TaskAlignedAssiger(nn.Module):
    def __init__(self,topk=10,num_classes=6,alpha=1.0,beta=6.0,eps=1e-9):
        super(TaskAlignedAssiger,self).__init__()
        
        self.topk = topk 
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta 
        self.eps = eps
                   
    @torch.no_grad()
    def forward(self,b_pd_scores,b_pd_bboxes,anc_points,b_gt_labels,b_gt_bboxes,b_gt_mask):
        self.bs,self.nc = b_pd_scores.shape[:2]
        # M : 标注数据对齐后的最大GT数目
        self.n_max_boxes = b_gt_bboxes.size(1) 
        
        target_bboxes,target_labels,fg_mask  = self.__get_targets(b_pd_scores,b_pd_bboxes,anc_points,b_gt_labels,b_gt_bboxes,b_gt_mask)
        
        return target_bboxes,target_labels,fg_mask
        
    def __get_targets(self,b_pd_scores,b_pd_bboxes,anc_points,b_gt_labels,b_gt_bboxes,b_gt_mask):
        # 记录正样本的类别 b x 8400 x cls_num
        b_target_labels = torch.zeros((self.bs,self.nc,self.num_classes),dtype=b_pd_scores.dtype,device=b_pd_scores.device)
        # 记录正样本的坐标 b x 8400 x 4
        b_target_bboxes = torch.zeros((self.bs,self.nc,4),dtype=b_pd_bboxes.dtype,device=b_pd_bboxes.device)
        # 记录8400个anchor正负样本的mask  b x  8400
        b_fg_mask = torch.zeros((self.bs,self.nc),dtype=torch.int64,device=b_pd_scores.device)
        # 记录预测与GTbox的ciou值
        # b_overlaps = torch.zeros([self.bs,self.n_max_boxes,self.nc],dtype=b_pd_bboxes.dtype,device=b_pd_bboxes.device)
        for i in range(self.bs):
            # 8400 x cls_num
            pb_scores = b_pd_scores[i]
            # 8400 x 4
            pb_bboxes = b_pd_bboxes[i]
            # M x 4
            gt_bboxes = b_gt_bboxes[i]
            # M x 1 
            gt_labels = b_gt_labels[i]
            # M x 1
            gt_mask = b_gt_mask[i]
            # ---------------------- 初筛正样本 ------------------------- #
            # -------------- 判断anchor锚点是否在gtbox内部 --------------- #
            # M x 8400
            in_gts_mask = self.__get_in_gts_mask(gt_bboxes,anc_points)
            # ---------------------- 精细筛选 ---------------------- #
            # 按照公式获取计算结果
            align_metrics,overlaps = self.__refine_select(pb_scores,pb_bboxes,gt_labels,gt_bboxes,in_gts_mask * gt_mask)
            
            # 根据计算结果,排序并选择top10
            # M x 8400 
            topk_mask = self.__select_topk_candidates(align_metrics,gt_mask.repeat(1,self.nc))
            # 本步可以去掉，为了保障，再过滤一遍
            pos_mask = topk_mask * in_gts_mask * gt_mask 

            # ------------------ 排除某个anchor被重复分配的问题 ---------------- #
            # target_gt_idx : 8400
            # fg_mask : 8400
            # pos_mask: M x 8400
            target_gt_idx,fg_mask,pos_mask = self.__filter_repeat_assign_candidates(pos_mask,overlaps)
            b_fg_mask[i] = fg_mask
            # 以上正样本的mask就完成了
            # ------------------ 根据Mask设置训练标签 ------------------ #
            # target_labels : 8400 x cls_num
            # target_bboxes : 8400 x 4
            target_labels,target_bboxes = self.__get_train_targets(gt_labels,gt_bboxes,target_gt_idx,fg_mask)
            b_target_bboxes[i] = target_bboxes
            
            # align_metric,overlaps均需要进行过滤
            align_metrics *= pos_mask # M x 8400 
            overlaps *= pos_mask # M x 8400
            
            # 找个每个GT的最大匹配值 M x 1
            gt_max_metrics = align_metrics.amax(axis=-1,keepdim=True)
            # 找到每个GT的最大CIOU值 M x 1
            gt_max_overlaps = overlaps.amax(axis=-1,keepdim=True)
            # 为类别one-hot标签添加惩罚项 M x 8400 -> 8400 -> 8400 x 1
            # 通过M个GT与所有anchor的匹配值 x 每个GT与所有anchor最大IOU / 每个类别与所有anchor最大的匹配值
            norm_align_metric = (align_metrics*gt_max_overlaps/(gt_max_metrics+self.eps)).amax(-2).unsqueeze(-1)
            # 8400 x cls_num
            target_labels = target_labels * norm_align_metric
            b_target_labels[i] = target_labels
        return b_target_bboxes,b_target_labels,b_fg_mask.bool()
    def __get_train_targets(self,gt_labels,gt_bboxes,target_gt_idx,fg_mask):
        '''
            gt_labels: M x 1 
            gt_bboxes: M x 4 
            fg_mask  : 8400 每个anchor为正负样本0或1
            target_gt_idx: 8400 每个anchor最匹配的GT索引(0~M)
        '''
        # gt_labels 拉直
        gt_labels = gt_labels.long().flatten()
        # 根据索引矩阵,获得cls  8400
        target_labels = gt_labels[target_gt_idx]
        # 同理bbox同样操作，
        # 根据索引矩阵，获得bbox 8400 x 4 
        target_bboxes = gt_bboxes[target_gt_idx]
        
        # 类别转换为one-hot形式，8400xcls_num
        target_one_hot_labels = torch.zeros((target_labels.shape[0],self.num_classes),
                                           dtype=torch.int64,
                                           device=target_labels.device)
        # 赋值，对应的类别位置置为1， 即one-hot形式
        target_one_hot_labels.scatter_(1,target_labels.unsqueeze(-1),1)
        
        # 生成对应的mask，用于过滤负样本 8400 -> 8400x1 -> 8400 x cls_num
        fg_labels_mask = fg_mask.unsqueeze(-1).repeat(1,self.num_classes)
        
        # 正负样本过滤
        target_one_hot_labels = torch.where(fg_labels_mask>0,target_one_hot_labels,0)
        
        return target_one_hot_labels,target_bboxes
            
            
    def __filter_repeat_assign_candidates(self,pos_mask,overlaps):
        '''
            pos_mask : M x 8400
            overlaps: M x 8400
            过滤原则:如某anchor被重复分配,则保留与anchor的ciou值最大的GT
        '''
        # 对列求和,即每个anchor对应的M个GT的mask值求和，如果大于1，则说明该anchor被多次分配给多个GT
        # 8400
        fg_mask = pos_mask.sum(0)
        if fg_mask.max() > 1:#某个anchor被重复分配
            # 找到被重复分配的anchor，mask位置设为True,复制M个，为了后面与overlaps shape匹配
            # 8400 -> 1 x 8400 -> M x 8400 
            mask_multi_gts = (fg_mask.unsqueeze(0) > 1).repeat([self.n_max_boxes, 1])
            # 每个anchor找到CIOU值最大的GT 索引  
            # 8400 
            max_overlaps_idx = overlaps.argmax(0)
            # 用于记录重复分配的anchor的与所有GTbox的CIOU最大的位置mask
            # M x 8400
            is_max_overlaps = torch.zeros(overlaps.shape, dtype=pos_mask.dtype, device=overlaps.device)
            # 每个anchor只保留ciou值最大的GT，对应位置设置为1
            is_max_overlaps.scatter_(0,max_overlaps_idx.unsqueeze(0),1)
            # 过滤掉重复匹配的情况
            pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask).float()
            # 得到更新后的每个anchor的mask 8400
            fg_mask = pos_mask.sum(0)
        # 找到每个anchor最匹配的GT 8400
        target_gt_idx = pos_mask.argmax(0)
        '''
            target_gt_idx: 8400 为每个anchor最匹配的GT索引(包含了正负样本)
            fg_mask: 8400 为每个anchor设置mask,用于区分正负样本
            pos_mask: M x 8400  每张图像中每个GT设置正负样本的mask
        '''
        return target_gt_idx,fg_mask,pos_mask
            
            
            
    def __select_topk_candidates(self,align_metric,gt_mask):
        # 从大到小排序,每个GT的从8400个结果中取前 topk个值，以及其中的对应索引
        # top_metrics : M x topk
        # top_idx : M x topk
        topk_metrics,topk_idx = torch.topk(align_metric,self.topk,dim=-1,largest=True)
        # 生成一个全0矩阵用于记录每个GT的topk的mask
        topk_mask = torch.zeros_like(align_metric,dtype=gt_mask.dtype,device=align_metric.device)
        for i in range(self.topk):
            top_i = topk_idx[:,i]
            # 对应的top_i位置值为1
            topk_mask[torch.arange(self.n_max_boxes),top_i] = 1
        topk_mask = topk_mask * gt_mask 
        # M x 8400
        return topk_mask 
            
            
            
    def __refine_select(self,pb_scores,pb_bboxes,gt_labels,gt_bboxes,gt_mask):
        # 根据论文公式进行计算得到对应的计算结果
        # reshape M x 4 -> M x 1 x 4 -> M x 8400 x 4 
        gt_bboxes = gt_bboxes.unsqueeze(1).repeat(1,self.nc,1)
        # reshape 8400 x 4 -> 1 x 8400 x 4 -> M x 8400 x 4 
        pb_bboxes = pb_bboxes.unsqueeze(0).repeat(self.n_max_boxes,1,1)
        # 计算所有预测box与所有gtbox的ciou，相当于公式中的U
        gt_pb_cious = bbox_iou(gt_bboxes,pb_bboxes,xywh=False,CIoU=True).squeeze(-1).clamp(0)
        # 过滤填充的GT以及不在GTbox范围内的部分
        # M x 8400
        gt_pb_cious = gt_pb_cious * gt_mask 

        # 获取与GT同类别的预测结果的scores 
        # 8400 x cls_num -> 1 x 8400 x cls_num -> M x 8400 x cls_num
        pb_scores = pb_scores.unsqueeze(0).repeat(self.n_max_boxes,1,1)
        # M x 1 -> M 
        gt_labels = gt_labels.long().squeeze(-1)
        # 针对每个GTBOX从预测值(Mx8400xcls_num)中筛选出对应自己类别Cls的结果,每个结果shape 1x8400
        # M x 8400 
        scores  = pb_scores[torch.arange(self.n_max_boxes),:,gt_labels]

        # 根据公式进行计算 M x 8400
        align_metric = scores.pow(self.alpha) * gt_pb_cious.pow(self.beta)
        # 过滤填充的GT以及不在GTbox范围内的部分
        align_metric = align_metric * gt_mask
        return align_metric,gt_pb_cious
        
        
    def __get_in_gts_mask(self,gt_bboxes,anc_points):
        # 找到M个GTBox的左上与右下坐标 M x 1 x 2
        gt_bboxes = gt_bboxes.view(-1,1,4)
        lt,rb = gt_bboxes[...,:2],gt_bboxes[...,2:]
        # anc_points 增加一个维度 1 x 8400 x 2
        anc_points = anc_points.view(1,-1,2)
        # 差值结果 M x 8400 x 4 
        bbox_detals = torch.cat([anc_points - lt,rb - anc_points],dim=-1)
        # 第三个维度均大于0才说明在gt内部
        # M x 8400
        in_gts_mask = bbox_detals.amin(2).gt_(self.eps)
        return in_gts_mask 
    