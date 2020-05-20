import torch
import torch.nn as nn
import scipy.spatial

class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        #dim=1按行求和    dismat是一个（batch_size，num_classes）的矩阵，每一行的200个数相同，代表一个样本的200的特征的组合，
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())#1 × dismat + (-2)×（self.centers.t() @ self.centers.t()）
        classes = torch.arange(self.num_classes).long()#tensor([ 0,  1,  2,  3......])
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#0代表第一维度，1代表第二维度    squeeze是去掉某个维度，unsqueeze是增加一个维度，不管是去掉还是增加的都是1。原来是（2,3），如果在第二维上增加就是（2,1,3）
        #label的样子是batch_size行，每一行的数字相同（是类别名）
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#对应位置比较是否相等，相等该位置为1，不等为0
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size#最小的是10的负12次方将dist中不在min~max之间的数值调整过来
        return loss