#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, margin_scale=1, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m_scale = margin_scale
        self.s = scale
        self.in_feats = num_out
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_class, num_out), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        self.nClasses = num_class

        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin

        print('Initialised adaptive-AAMSoftmax margin scale %.3f scale %.3f'%(self.m,self.s))
    def calc_m(self, s_n):
        avg_s_n = torch.mean(s_n)
        m = torch.arccos(avg_s_n) * self.m_scale
        return m
    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        N = x.size()[0]

        shifts = -1 * label.unsqueeze(-1)
        arange1 = torch.arange(self.nClasses).view((1, self.nClasses)).repeat((N,1)).cuda()
        arange2 = (arange1 - shifts) % self.nClasses

        w_norm = F.normalize(self.weight)

        x_norm = F.normalize(x)

        s_n = []

        costh_x = torch.mm(x_norm, x_norm.transpose(0,1))
        s_n.append(costh_x.flatten()[1:].view(N-1, N+1)[:,:-1].transpose(0,1).flatten()[0:N*(N-1)//2])
        cosine = torch.mm(x_norm, w_norm)
        cosine2 = torch.gather(cosine, 1, arange2)            
        s_n.append(cosine2[:,1:].flatten())


        costh_w = torch.mm(w_norm.transpose(0,1), w_norm) # (n_class, n_class)
        s_n.append(costh_w.flatten()[1:].view(self.nClasses-1, self.nClasses+1)[:,:-1].transpose(0,1).flatten()[0:self.nClasses*(self.nClasses-1)//2])
        s_n = torch.cat(s_n)            

        m = self.calc_m(s_n)

        cos_m = math.cos(m)
        sin_m = math.sin(m)

        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
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
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        
        return loss, prec1, prec1, 0