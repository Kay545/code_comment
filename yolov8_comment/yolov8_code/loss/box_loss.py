import torch 
import torch.nn as nn             
from utils.utils import *
import torch.nn.functional as F
class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super(BboxLoss,self).__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = self.bbox2reg(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target): 
        """Return sum of left and right DFL losses."""
        # target向下取整，作为目标的左侧整数,作为目标的左侧整数 K x 4（取值范围0-15）,相当于公式中的Si
        tl = target.long()  # target left  (包括中心点距离左边和上边的距离)
        # tl加上整数1，作为目标的右侧整数， K x 4（取值范围0-15）,相当于公式中的Si+1
        tr = tl + 1  # target right
        # 分别将偏移值作为权重,如真实坐标值为8.3，那么它距离8近给一个大一点的权重0.7，距离9远，给一个小一点的权重0.3
        wl = tr - target  # weight left Kx4
               
        wr = target - tl  # weight right Kx4
        # 左右目标分别拉直 tl -> K x 4 -> 4K , pred_dist->4K x 16
        l_loss = F.cross_entropy(pred_dist, tl.view(-1), reduction='none')
        # 左右目标分别拉直 tr -> K x 4 -> 4K , pred_dist->4K x 16
        r_loss = F.cross_entropy(pred_dist, tr.view(-1), reduction='none')
        loss = l_loss.view(tl.shape) * wl + r_loss.view(tl.shape) * wr                
        loss = loss.mean(-1,keepdim=True)

        return loss 

    def bbox2reg(self,anchor_points, target_bboxes,reg_max):

        """Transform bbox(xyxy) to dist(ltrb)."""
        x1y1,x2y2 = target_bboxes[...,:2],target_bboxes[...,2:]
        return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)