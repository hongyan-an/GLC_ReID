#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from star.src.models.utils import GraphConv, MeanAggregator
import datetime
import torch.optim as optim
import random
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class GCN(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        #self.conv2 = GraphConv(nhid, nhid, MeanAggregator, dropout)
        #self.conv3 = GraphConv(nhid, nhid, MeanAggregator, dropout)

    def forward(self, data):
        x, adj = data[0], data[1]

        x = self.conv1(x, adj)
        #x = F.relu(self.conv2(x, adj)+x)
        #x = F.relu(self.conv3(x, adj) + x)

        return x

class HEAD(nn.Module):
    def __init__(self, nhid, dropout=0,loss='CE'):
        super(HEAD, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(nn.Linear(nhid*2, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, 2))
        if loss=='CE':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss=FocalLoss(2)

    def forward(self, feature, data, select_id=None):
        adj, label = data[1], data[2]
        feature = feature.view(-1,self.nhid)
        inst = adj._indices().size()[1]
        select_id = select_id

        if select_id is None:
            print('Dont have to select id.')
            row = adj._indices()[0,:]
            col = adj._indices()[1,:]
        else:
            row = adj._indices()[0, select_id].tolist()
            col = adj._indices()[1, select_id].tolist()
        patch_label = (label[row] == label[col]).long()
        pred = self.classifier(torch.cat((feature[row],feature[col]),1))

        loss = self.loss(pred, patch_label)
        return loss 

class HEAD_test(nn.Module):
    def __init__(self, nhid):
        super(HEAD_test, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(nn.Linear(nhid*2, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, 2),nn.Softmax(dim=1))

    def forward(self, feature1, feature2, no_list=False):
        if len(feature1.size())==1:
            pred = self.classifier(torch.cat((feature1,feature2)).unsqueeze(0))
            if pred[0][0]>pred[0][1]:
                is_same = False
            else:
                is_same = True
            return is_same
        else:
            pred = self.classifier(torch.cat((feature1,feature2),1))
            #print(pred[0:10,:])
            if no_list:
                return pred[:,1]
            score = list(pred[:,1].cpu().detach().numpy())
            #is_same = (pred[:,0]<pred[:,1]).long()

        return score


def gcn(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
