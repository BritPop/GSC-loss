#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, scale=30.0, hard_n = 20000, hard_d = 0, hard_p = 0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.s = scale
        self.in_feats = num_out
        self.W = torch.nn.Parameter(torch.randn(num_out, num_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        self.nClasses = num_class
        self.K = hard_n
        self.d = hard_d
        self.k = hard_p
        print('Initialised Generalized Score Comparison Loss hard_N %d hard_P %d hard_d %.3f'%(self.K, self.k, self.d))

    def forward(self, x, label=None):
        if len(x.size())==2:
            x= x.unsqueeze(1)
        N, M, _ = x.size()
        shifts = -1 * label.unsqueeze(-1)
        arange1 = torch.arange(self.nClasses).view((1, self.nClasses)).repeat((N,1)).cuda()
        arange2 = (arange1 - shifts) % self.nClasses
        
        w_norm = torch.div(self.W, torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12))
        x_norm = []
        for i in range(M):
            x_norm.append(torch.div(x[:, i, :], torch.norm(x[:, i, :], p=2,dim=1, keepdim=True).clamp(min=1e-12)))

        s_n = []
        s_p = []
        for i in range(M):
            costh_x = torch.mm(x_norm[i], x_norm[i].transpose(0,1))
            s_n.append(costh_x.flatten()[1:].view(N-1, N+1)[:,:-1].transpose(0,1).flatten()[0:N*(N-1)//2])
            costh_x_w = torch.mm(x_norm[i], w_norm)
            costh2_x_w = torch.gather(costh_x_w, 1, arange2)            
            s_p.append(costh2_x_w[:,0])
            s_n.append(costh2_x_w[:,1:].flatten())
            for j in range(M):
                if(j>i):
                    costh_x_x = torch.mm(x_norm[i],x_norm[j].transpose(0,1))
                    s_p.append(costh_x_x.diagonal())
                    s_n.append(costh_x_x.flatten()[1:].view(N-1, N+1)[:,:-1].flatten())
                    
        costh_w = torch.mm(w_norm.transpose(0,1), w_norm) # (n_class, n_class)
        s_n.append(costh_w.flatten()[1:].view(self.nClasses-1, self.nClasses+1)[:,:-1].transpose(0,1).flatten()[0:self.nClasses*(self.nClasses-1)//2])    
        
        s_n = torch.cat(s_n)            
        s_p = torch.cat(s_p)
            
        s_n, _ = s_n.topk(self.K)
        min_s_p = torch.min(s_p)
        check = torch.where(s_n>min_s_p-self.d, True, False)
        s_n = torch.masked_select(s_n, check)
        
        if self.k != 0:
            s_p, _ = s_p.topk(self.k, largest=False)
        
        n_n = s_n.size()[0]
        n_p = s_p.size()[0]
        
        s_p = s_p.unsqueeze(-1) 
        s_n = s_n.unsqueeze(0).repeat(n_p, 1) 
        
        s = torch.cat([s_p, s_n], dim=-1)
        
        cos_sim_matrix = s*self.s
        
        label1 = torch.from_numpy(numpy.zeros(n_p).astype(int)).cuda()
        
        costh_x_w = self.s * costh_x_w

        loss   = self.ce(cos_sim_matrix, label1)
        #prec1   = accuracy(costh_x_w.detach(), label.detach(), topk=(1,))[0]
        prec2   = accuracy(cos_sim_matrix.detach(), label1.detach(), topk=(1,))[0]
            
        return loss, prec2, prec2, n_n