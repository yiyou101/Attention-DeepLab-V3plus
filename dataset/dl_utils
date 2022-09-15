import torch.utils.data as D
import torchvision
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
import glob
import cv2
from image_utils import readTif,readTif_info

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class openDataset(Dataset):
    def __init__(self, datapath, labelpath, activation_fn):
        if isinstance(datapath,list):
            self.datalist = datapath
            self.labellist = labelpath
        else:
            self.datalist = glob.glob(datapath+'/*.tif')
            self.labellist = glob.glob(labelpath+'/*.tif')
        self.len = len(self.datalist)
        self.activation_fn = activation_fn
        self.tans = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), ])
    def __getitem__(self, item):
        data = readTif(self.datalist[item])
        #data = data.swapaxes(1,0).swapaxes(1,2)
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
        #data = self.tans(data)
        label = readTif(self.labellist[item])
        #one_hot = torch.nn.functional.one_hot(label)
        if self.activation_fn == 'sigmoid':
            label = np.array([label])
            return result, label.astype(float)
        else:
            return result, label.astype(np.int64)
    def __len__(self):
        return self.len

class test_openDataset(Dataset):
    def __init__(self, datapath):
        self.datalist = glob.glob(datapath+'/*.tif')
        self.len = len(self.datalist)
    def __getitem__(self, item):
        data, trans, proj = readTif_info(self.datalist[item])
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
        return result, trans, proj
    def __len__(self):
        return self.len

@torch.no_grad()
def Binary_validation(model, loader):
    val_iou = []
    model.eval()
    activation_fn = nn.Sigmoid()
    for i,(image, target) in enumerate(loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        pred = model(image)
        pred = activation_fn(pred)
        N = target.size()[0]
        input_flat = pred.view(N, -1)
        targets_flat = target.view(N, -1)
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        iou = dice_eff.sum() / N
        val_iou.append(iou.item())
    iou_num = np.mean(val_iou)
    return iou_num
    
@torch.no_grad()
def validation(model, loader):
    val_iou = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        softmax = nn.Softmax()
        output = softmax(output)
        output = output.argmax(1)
        iou = cal_iou(output, target)/output.shape(0)
        val_iou.append(iou)
    return val_iou

def cal_iou(pred, target, class_num):
    all_iou = 0
    for idx in range(class_num):
        N = target.size()[0]
        smooth = 1
        p = (pred == idx).int().reshape(-1)
        t = (target == idx).int().reshape(-1)
        union = p.sum() + t.sum()
        overlap = (p * t).sum()
        # print(idx, uion, overlap)
        iou = overlap / (union + smooth - overlap)
        all_iou = all_iou + iou
    miou = all_iou/(class_num*N)
    return miou

@torch.no_grad()
def validation_loss(model, loader, loss):
    loss_list = []
    model.eval()
    for i ,(image, target) in enumerate(loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        loss_list.append(loss(output,target).item())
    loss_num = np.mean(loss_list)
    return loss_num

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.activation_fn = nn.Softmax()

    def forward(self, pred, target):
        pred = self.activation_fn(pred)
        N = target.size()[0]
        class_num = pred.shape[1]
        pred = pred.argmax(1)
        all_dice = 0
        for idx in range(class_num):
            smooth = 1
            p = (pred == idx).int().reshape(-1)
            t = (target == idx).int().reshape(-1)
            union = p.sum() + t.sum()
            overlap = (p * t).sum()
            # print(idx, uion, overlap)
            dice = 2 * overlap / (union + smooth)
            all_dice = all_dice + dice
        diceloss = 1 - all_dice / (N * class_num)
        return diceloss

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.activation_fn = nn.Sigmoid()

    def forward(self, pred, target):
        pred = self.activation_fn(pred)
        N = target.size()[0]
        input_flat = pred.view(N, -1)
        targets_flat = target.view(N, -1)
        # 计算交集
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - dice_eff.sum() / N
        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        """
                :param alpha_t: A list of weights for each class
                :param gamma:
        """
        self.alpha = torch.tensor(alpha) if alpha else None
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs)
        CE = torch.nn.CrossEntropyLoss()
        CE_loss = CE(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        if self.reduce == 'avg':
            return F_loss.mean()
        else:
            return F_loss.sum()

class loss_joint(nn.Module):
    def __init__(self, loss_fn1, loss_fn2, weight):
        super(loss_joint, self).__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight = weight

    def forward(self, pred, target):
        loss1 = self.loss_fn1(pred, target)
        loss2 = self.loss_fn2(pred, target)
        loss = self.weight[0]*loss1 + self.weight[1]*loss2
        return loss
