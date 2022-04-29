import imutils
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import scipy.misc as misc
import cutting
import Constants
import random
import nrrd
import albumentations as A
from sklearn.model_selection import KFold,RepeatedKFold,RepeatedStratifiedKFold

def default_Colon_loader(train_ctimg_path, train_mriimg_path,train_label_path):

    train_ctimg=train_ctimg_path
    train_mriimg=train_mriimg_path
    train_label=train_label_path

    return train_ctimg, train_mriimg,train_label


def read_Colon_datasets(root_path, aa):

    pcr_label = []
    train_img1=[]
    train_label1=[]
    test_img1=[]
    test_label1=[]

    pcr_image_root = root_path
    pcr_ctimage_root=pcr_image_root+'/'
    pcr_mriimage_root=pcr_image_root+'/'


    pcr_label_path = 'E:/Dataset/.csv'


    pcr_ctfile = os.listdir(pcr_ctimage_root)
    pcr_mrifile = os.listdir(pcr_mriimage_root)
    pcr_ctfile.sort(key=lambda x: str(x[:-6]))
    pcr_mrifile.sort(key=lambda x: str(x[:-6]))
    t_ctimg=[]
    t_mriimg=[]
    t_label=[]
    v_ctimg=[]
    v_mriimg=[]
    v_label=[]

    with open(pcr_label_path, 'r') as f1:
        pcr_label1 = f1.read().split('\n')

    for i in range(len(pcr_label1) - 1):
        a = pcr_label1[i].split(',')[-1]
        a = int(a)
        pcr_label.append(a)

    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=10)
    for train_index,test_index in kf.split(pcr_ctfile, pcr_label):
        train_img1.append(train_index)
        train_label1.append(train_index)
        test_img1.append(test_index)
        test_label1.append(test_index)

    train_img=train_img1[aa]
    train_label=train_label1[aa]
    test_img=test_img1[aa]
    test_label=test_label1[aa]
    for i in range(len(train_img)):
        ctname=pcr_ctimage_root+pcr_ctfile[train_img[i]]+'/'
        mriname=pcr_mriimage_root+pcr_mrifile[train_img[i]]+'/'
        ctimg_name=os.listdir(ctname)
        mriimg_name=os.listdir(mriname)
        asi = len(ctimg_name)
        bsi = len(mriimg_name)
        if(asi>=bsi):
            zsi=bsi
        else:
            zsi=asi
        for ii in range(zsi):
            ra = random.randint(10, 180)
            ra1 = random.randint(190, 350)

            cv2_ctname=ctname+ctimg_name[ii]
            cv2_mriname=mriname + mriimg_name[ii]
            ctimg=cv2.imread(cv2_ctname)
            mriimg=cv2.imread(cv2_mriname)
            t_ctimg.append(ctimg)
            t_mriimg.append(mriimg)
            t_label.append(pcr_label[train_label[i]])


            if pcr_label[train_label[i]] == 1:
                img1 = imutils.rotate(ctimg, ra)
                img11 = imutils.rotate(mriimg, ra)
                t_ctimg.append(img1)
                t_mriimg.append(img11)
                t_label.append(int(1))
                img2 = imutils.rotate(ctimg, ra1)
                img22 = imutils.rotate(mriimg, ra1)
                t_ctimg.append(img2)
                t_mriimg.append(img22)
                t_label.append(int(1))
                img3 = cv2.flip(ctimg, 1)  # 水平翻转
                img33 = cv2.flip(mriimg, 1)  # 水平翻转
                t_ctimg.append(img3)
                t_mriimg.append(img33)
                t_label.append(int(1))
                img4 = cv2.flip(ctimg, 0)  # 垂直翻转
                img44 = cv2.flip(mriimg, 0)  # 垂直翻转
                t_ctimg.append(img4)
                t_mriimg.append(img44)
                t_label.append(int(1))
                img5 = cv2.flip(ctimg, -1)  # 水平垂直翻转
                img55 = cv2.flip(mriimg, -1)  # 水平垂直翻转
                t_ctimg.append(img5)
                t_mriimg.append(img55)
                t_label.append(int(1))

    return t_ctimg,t_mriimg,t_label




class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='Colon', aa=0):
        self.root = root_path
        self.aa = aa
        self.dataset = datasets

        if self.dataset == 'Colon':
            self.train_ctimg,self.train_mriimg,self.train_label = read_Colon_datasets(self.root, self.aa)


    def __getitem__(self, index):

        train_ctimg, train_mriimg,train_label = default_Colon_loader(self.train_ctimg[index],self.train_mriimg[index],self.train_label[index])

        tran = A.Compose([ A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_ctimg = tran(image=train_ctimg)
        train_ctimg = np.transpose(train_ctimg['image'], (2, 0, 1))
        train_ctimg = torch.Tensor(train_ctimg).cuda()

        train_mriimg = tran(image=train_mriimg)
        train_mriimg = np.transpose(train_mriimg['image'], (2, 0, 1))
        train_mriimg = torch.Tensor(train_mriimg).cuda()

        train_label = torch.tensor(train_label).cuda()

        return train_ctimg,train_mriimg, train_label

    def __len__(self):
        assert len(self.train_ctimg) == len(self.train_label), 'The number of images must be equal to labels'
        return len(self.train_ctimg)
