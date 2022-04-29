import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=1e-3, evalmode=False):
        self.net = net()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred= self.net.forward(self.img)
        loss, Dice, IoU = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred,Dice,IoU
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path),strict=False)
    
    def paraNum(self):
        print("the network have {} paramerters in total".format(sum(x.numel() for x in self.net.parameters())))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        mylog.write( 'update learning rate: %f -> %f\n' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
