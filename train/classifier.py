import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os
from pytorchtools import EarlyStopping

from Network import ClassifierNet
from classfier_data import ImageFolder

import pandas as pd

index=15
batch_size = 16
train_dataset = ImageFolder(root_path= 'E:/Dataset/', datasets='Colon',aa=index)  # 'Colon_p'
train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
val_dataset = ImageFolder(root_path= 'E:/Dataset/', datasets='Colon',aa=index)  # 'Colon_p'
val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)


net = ClassifierNet.Net().cuda()
loss_func = nn.BCELoss()

optimizer = optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=10, verbose=True)

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    v_id=[]
    v_pre=[]
    v_lab=[]
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        valid_data_size=len(train_data_loader)
        train_data_size=len(val_data_loader)
        TP=0
        TN=0
        FP=0
        FN=0

        for i, (ctinputs,mriinputs, labels) in enumerate(train_data_loader):

            ctinputs = ctinputs.to(device)
            mriinputs = mriinputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(ctinputs,mriinputs)
            outputs=outputs.squeeze()
            labels_1=labels.float()

            loss = loss_function(outputs, labels_1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions=outputs
            label = labels.cpu().numpy()
            cls = predictions.cpu().detach().numpy()
            ba=len(label)

            for i in range(ba):
                if (label[i] == 0) & (cls[i] < 0.5):
                    TN = TN + 1
                if (label[i] == 0) & (cls[i] >= 0.5):
                    FP = FP + 1
                if (label[i] == 1) & (cls[i] >= 0.5):
                    TP = TP + 1
                if (label[i] == 1) & (cls[i] < 0.5):
                    FN = FN + 1



        Acc = (TP + TN) / (TP + FN + FP + TN + 0.01)
        Precision = TP / (TP + FP + 0.01)
        Sen = TP / (TP + FN + 0.01)
        Spe = TN / (TN + FP + 0.01)


        with torch.no_grad():
            model.eval()
            TP1=0
            TN1=0
            FP1=0
            FN1=0
            for j, (val_ctinputs,val_mriinputs,val_labels,id) in enumerate(val_data_loader):
                if(len(val_labels)==1):
                    break
                val_ctinputs = val_ctinputs.to(device)
                val_mriinputs = val_mriinputs.to(device)
                labels = val_labels.to(device)

                outputs = model(val_ctinputs,val_mriinputs)
                outputs = outputs.squeeze()
                labels_1 = labels.float()

                loss = loss_function(outputs, labels_1)

                valid_loss += loss.item()

                predictions=outputs

                label1 = labels.cpu().numpy()
                cls1 = predictions.cpu().detach().numpy()

                ba1=len(label1)
                for ii in range(ba1):
                    if (label1[ii] == 0) & (cls1[ii] <0.5):
                        TN1 = TN1 + 1
                    if (label1[ii] == 0) & (cls1[ii] >=0.5):
                        FP1 = FP1 + 1
                    if (label1[ii] == 1) & (cls1[ii] >=0.5):
                        TP1 = TP1 + 1
                    if (label1[ii] == 1) & (cls1[ii] <0.5):
                        FN1 = FN1 + 1

            Acc1 = (TP1 + TN1) / (TP1 + FN1 + FP1 + TN1 + 0.01)
            Precision1 = TP1 / (TP1 + FP1 + 0.01)
            Sen1 = TP1 / (TP1 + FN1 + 0.01)
            Spe1 = TN1 / (TN1 + FP1 + 0.01)


        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, Acc, Acc1])

        if best_acc < Acc1:
            best_acc = Acc1
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Acc: {:.4f}%,Precision: {:.4f}%,Sen: {:.4f}%,Spe: {:.4f}% \n\t\tValidation: Loss: {:.4f}, Acc: {:.4f}%,Precision: {:.4f}%,Sen: {:.4f}%,Spe: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss,  Acc * 100, Precision* 100,Sen* 100,Spe* 100,avg_valid_loss, Acc1 * 100, Precision1* 100,Sen1* 100,Spe1* 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        print('train')
        print('TP:',TP)
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('Val')
        print('TP:',TP1)
        print('TN:',TN1)
        print('FP:',FP1)
        print('FN:',FN1)
        # early_stopping = EarlyStopping(patience=5, verbose=True)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        #torch.save(model, 'models/' + '177' + '_model_' + str(epoch + 1) + '.pt')
    return model, history ,v_id,v_lab,v_pre


num_epochs = 100
trained_model, history ,v_id,v_lab,v_pre= train_and_valid(net, loss_func,optimizer, num_epochs)
#torch.save(history, 'models/' + '99' + '_history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('99' + '_loss_curve.png')
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('99' + '_accuracy_curve.png')
plt.show()

