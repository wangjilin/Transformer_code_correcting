
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class LabelSmoothing(nn.Module):
    "Implement label smoothing. size表示类别总数 "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing#if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()#先深复制过来
        print(true_dist)
        #print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))#otherwise的公式
        print(true_dist)
        #print true_dist
        #变成one-hot编码，1表示按列填充，
        #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        print(true_dist)
        self.true_dist = true_dist
        print(x.shape,true_dist.shape)
        return self.criterion(x, Variable(true_dist, requires_grad=False))


crit = LabelSmoothing(size=5,smoothing= 0.1)
#predict.shape 3 5
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.9, 0.2, 0.1, 0],
                            [1, 0.2, 0.7, 0.1, 0]])

v = crit(Variable(predict.log()),Variable(torch.LongTensor([2, 1, 0])))
 
print(predict.log().size(),torch.LongTensor([2, 1, 0]).size())