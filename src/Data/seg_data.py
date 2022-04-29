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
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask
def default_Colon_loader(img_path, mask_path,id):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    id = id
    return img, mask, id



def read_Colon_datasets(root_path, aa):

    pcr_label = []
    train_img1=[]
    train_label1=[]
    test_img1=[]
    test_label1=[]

    image_root = root_path
    image_data=image_root+'data/'
    image_label=image_root+'label/'

    pcr_label_path="E:/Dataset/.csv"

    datafile = os.listdir(image_data)
    labelfile = os.listdir(image_label)
    datafile.sort(key=lambda x: str(x[:-6]))
    labelfile.sort(key=lambda x: str(x[:-6]))
    t_dataimg=[]
    t_labelimg=[]
    t_label=[]
    v_dataimg=[]
    v_labelimg=[]
    v_label=[]
    id=[]




    with open(pcr_label_path, 'r') as f1:
        pcr_label1 = f1.read().split('\n')

    for i in range(len(pcr_label1) - 1):
        a = pcr_label1[i].split(',')[-1]
        a = int(a)
        pcr_label.append(a)

    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=10)
    for train_index,test_index in kf.split(datafile, pcr_label):
        train_img1.append(train_index)
        train_label1.append(train_index)
        test_img1.append(test_index)
        test_label1.append(test_index)

    train_img=train_img1[aa]
    train_label=train_label1[aa]
    test_img=test_img1[aa]
    test_label=test_label1[aa]
    for i in range(len(train_img)):
        dataname=image_data+datafile[train_img[i]]+'/'
        labelname=image_label+labelfile[train_label[i]]+'/'
        dataimg_name=os.listdir(dataname)
        labelimg_name=os.listdir(labelname)

        for ii in range(len(dataimg_name)):
            cv2_dataname=dataname+dataimg_name[ii]
            cv2_labelname=labelname +labelimg_name[ii]
            t_dataimg.append(cv2_dataname)
            t_labelimg.append(cv2_labelname)
            id.append(labelfile[train_label[i]]+labelimg_name[ii])

    return t_dataimg,t_labelimg,id




class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='Colon', aa=0):
        self.root = root_path
        self.aa = aa
        self.dataset = datasets

        if self.dataset == 'Colon':
            self.train_dataimg,self.train_labelimg,self.id = read_Colon_datasets(self.root, self.aa)


    def __getitem__(self, index):

        train_dataimg, train_labelimg,id= default_Colon_loader(self.train_dataimg[index],self.train_labelimg[index],self.id[index])

        train_dataimg = torch.Tensor(train_dataimg)
        train_labelimg = torch.Tensor(train_labelimg)

        return train_dataimg,train_labelimg,id

    def __len__(self):
        assert len(self.train_dataimg) == len(self.train_labelimg), 'The number of images must be equal to labels'
        return len(self.train_dataimg)
