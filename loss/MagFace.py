#! /usr/bin/python
# -*- encoding: utf-8 -*-
# adapted from https://github.com/IrvingMeng/MagFace/blob/main/models/magface.py


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(
        self, 
        num_out, 
        num_class,  
        scale=30,
        easy_margin=False, 
        n_u=110.0,
        n_l=10.0,
        m_u=1.0,
        m_l=0.1, 
        lambda_g = 35,
        **kwargs
    ):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.in_feats = num_out
        self.out_features = num_class
        self.s = scale
        self.n_u = n_u
        self.n_l = n_l
        self.m_u = m_u
        self.m_l = m_l

        self.weight = torch.nn.Parameter(torch.FloatTensor(num_class, num_out), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        
        self.ce = nn.CrossEntropyLoss()
        
        self.lambda_g = lambda_g
        print('Initialised MagFace scale %.3f n_u %.3f n_l %.3f m_u %d m_l %d l_g %.3f '%(self.s,self.n_u,self.n_l,self.m_u,self.m_l, self.lambda_g))

    def calc_loss_G(self, x_norm):
        g = 1/(self.n_u**2)*x_norm + 1/(x_norm)
        return torch.mean(g)
    def calc_m(self, x):
        margin = (self.m_u-self.m_l)/(self.n_u-self.n_l)*(x-self.n_l) + self.m_l
        return margin
    
    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.n_l, self.n_u)
        ada_margin = self.calc_m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        th = torch.cos(math.pi - ada_margin)
        mm = torch.sin(math.pi - ada_margin) * ada_margin
        cosine = F.linear(
            F.normalize(x), F.normalize(self.weight)
        )
        # (batch, classes)


        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - th) > 0, phi, cosine - mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)

        loss_g = self.calc_loss_G(x_norm)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss + self.lambda_g*loss_g, prec1, prec1, 0