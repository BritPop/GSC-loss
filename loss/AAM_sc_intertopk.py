# code from WeSpeaker: https://github.com/wenet-e2e/wespeaker/blob/
# Adapted from https://github.com/espnet/espnet/blob/master/espnet2/spk/loss/aamsoftmax_subcenter_intertopk.py
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
        margin=0.3, 
        scale=15, 
        easy_margin=False, 
        K_sc=3, 
        mp=0.06, 
        k_top=5, 
        do_lm=False, 
        **kwargs
    ):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.in_feats = num_out
        self.out_features = num_class
        self.s = scale
        self.margin = margin
        self.do_lim = do_lm

        self.K_sc = K_sc
        # intertopk + subcenter
        if do_lm:
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top
        
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.K_sc*num_class, num_out), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        self.m = self.margin

        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)
        
        self.ce = nn.CrossEntropyLoss()
        print('Initialised AAMSoftmax+subcenter+intertopk margin %.3f scale %.3f K(num_sub-center) %d k_top %d'%(self.m,self.s,self.K_sc, self.k_top))

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        cosine = F.linear(
            F.normalize(x), F.normalize(self.weight)
        )
        # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K_sc)
        )  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp


        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, (label).view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = x.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi)
                + (top_k_one_hot * phi_mp)
                + ((1.0 - one_hot - top_k_one_hot) * cosine)
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1, prec1, 0