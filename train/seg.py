import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from Network.seg import CE_Net_
from framework import MyFrame
import loss

from seg_data import ImageFolder

import random
import Constants

savepath='./results/'
import numpy as np

# Please specify the ID of graphics cards that you want to use
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def Train_net():
    NAME = 'CE' + Constants.ROOT.split('/')[-1]


    solver = MyFrame(CE_Net_, loss.dice_bce_loss, 1e-3)
    batchsize =16

    dataset = ImageFolder(root_path="E:/Dataset/", datasets='Colon',aa=15)#'Colon_p'
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    # print the total number of parameters in the network
    solver.paraNum()

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = 100
    train_epoch_best_loss = 10000
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        Dice=0
        IOU=0

        for img, mask ,img_id in data_loader_iter:

            solver.set_input(img, mask)
            train_loss, pred,dice1,iou1 = solver.optimize()

            train_epoch_loss += train_loss
            Dice += dice1
            IOU += iou1
            index = index + 1

            for i in range(len(img_id)):
                pre11=np.transpose(pred[i, :, :, :].cpu().detach().numpy() , (1, 2, 0))
                cv2.imwrite(savepath + img_id[i],pre11*255)

        print('epoch:', epoch, '    time before imwrite:', int(time() - tic))

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        Dice = Dice/len(data_loader_iter)
        IOU = IOU/len(data_loader_iter)
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('totalNum in an epoch:',index)
        print('train_loss:', train_epoch_loss)
        print('Dice:',  Dice)
        print('IOU:', IOU)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/1/' + NAME + '_plus_spatial_multi.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-10:
                break
            solver.load('./weights/1/' + NAME + '_plus_spatial_multi.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    Train_net()



