#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, pdb, sys, random, time, os, itertools, shutil, importlib
import numpy as np
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
#from hyperion.hyperion.score_norm import AdaptSNorm

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, num_utt, **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)
        self.num_utt = num_utt
        self.num_pooling = kwargs.pop('num_pooling')
        self.dur_grouping = kwargs.pop('dur_grouping')

    def forward(self, data, label=None):
        if label == None:
            return self.__S__.forward(data.reshape(-1,data.size()[-1]).cuda(), aug=False) # from exp07
        else:
            if self.dur_grouping:
                outp = []
                for d in data: # 2022.03.16
                    outp += [self.__S__.forward(d.cuda(), aug=True)]
                outp = torch.stack(outp).transpose(0,1)                
            else:
                data = data.reshape(-1, data.size()[-1]).cuda() # from exp07
                outp = self.__S__.forward(data, aug=True)
                outp = outp.reshape(self.num_utt, -1, outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1, prec2, n_n = self.__L__.forward(outp, label)
            return nloss, prec1, prec2, n_n

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):
        self.__model__  = speaker_model
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__ = Scheduler(self.__optimizer__, **kwargs)
        self.scaler = GradScaler() 
        self.gpu = gpu
        self.ngpu = int(torch.cuda.device_count())
        self.ndistfactor = int(kwargs.pop('num_utt') * self.ngpu)
        self.dur_grouping = kwargs.pop('dur_grouping')
        #self.asnorm = AdaptSNorm(nbest=10)

    def train_network(self, loader, epoch, verbose):
        self.__model__.train()
        self.__scheduler__.step(epoch-1)
        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss, top1, top2, N_N = 0, 0, 0, 0, 0, 0
        tstart = time.time()
        for data, data_label in loader:
            self.__model__.zero_grad()

            if self.dur_grouping: # 2022.03.16
                data = self.grouping(data)
            else:
                data = data.transpose(1,0) # from exp07
            label = torch.LongTensor(data_label).cuda()

            with autocast():
                nloss, prec1, prec2, n_n = self.__model__(data, label)
            self.scaler.scale(nloss).backward()
            self.scaler.step(self.__optimizer__)
            self.scaler.update()

            loss    += nloss.detach().cpu().item()
            top1    += prec1.detach().cpu().item()
            top2    += prec2.detach().cpu().item()
            N_N     += n_n
            cnt += 1
            idx     += bs
            lr = self.__optimizer__.param_groups[0]['lr']
            telapsed = time.time() - tstart
            tstart = time.time()
            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}: Loss {:f}, ACC1 {:2.3f}%, ACC2 {:2.3f}%, N_N {:.2f} LR {:.6f} - {:.2f} Hz ".format(idx*df, loader.__len__()*bs*df, loss/cnt, top1/cnt, top2/cnt, N_N/cnt, lr, bs*df/telapsed))
                sys.stdout.flush()
        return (loss/cnt, top1/cnt, top2/cnt, N_N/cnt, lr)

    def grouping(self, data): # from exp14
        dr = [150, 300]
        data_g = []
        for i in range(data.size()[1]):
            data_g += [data[:, i, 0:(dr[i] * 160)]] #+ 240
        return data_g

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
            
    def evaluateFromList(self, test_list, test_path, train_list, train_path, num_thread, distributed, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()
        ## Eval loader ##
        feats_eval  = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files    = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=0, num_eval=1, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            audio     = data[0][0].cuda()
            with torch.no_grad():
                ref_feat_1 = self.__model__(audio).detach().cpu() # full
            feats_eval[data[1][0]] = ref_feat_1.unsqueeze(0)
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat_1.size()[1]))
                sys.stdout.flush()


        ## Compute verification scores ##
        all_scores_1, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat_1 = feats_eval[data[1]][0].cuda() # enr - full
                tst_feat_1 = feats_eval[data[2]][0].cuda() # tst - full
                if self.__model__.module.__L__.test_normalize:

                    enr_feat_1 = F.normalize(enr_feat_1, p=2, dim=1)
                    tst_feat_1 = F.normalize(tst_feat_1, p=2, dim=1)

                score_1 = F.cosine_similarity(enr_feat_1, tst_feat_1)
                
                all_scores_1.append(score_1.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores_1, all_labels)
