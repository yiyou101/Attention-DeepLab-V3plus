import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision
from osgeo import gdal
from torch.utils.data import Dataset,DataLoader
import cv2
from torch.utils.tensorboard import SummaryWriter
from dl_utils import BinaryDiceLoss,openDataset,loss_joint,split_train_val,Binary_validation,validation_loss
import Attention DeepLab 


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(train_img, train_label, opt_name, train_epoch, model_path, last_model):
    writer = SummaryWriter()
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(train_img, train_label, 0.9)
    dataset = openDataset(train_image_paths, train_label_paths, 'sigmoid')
    val_data = openDataset(val_image_paths, val_label_paths, 'sigmoid')
    train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=3, shuffle=True, num_workers=0)
    model = Attention DeepLab.BNDDeepLab(12,1)
    if opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=2,  
        T_mult=2,  
        eta_min=1e-6  
    )
    model.to(device)
    diceloss = BinaryDiceLoss()
    BCE = torch.nn.BCEWithLogitsLoss()
    Binary_criterion = loss_joint(diceloss, BCE, [0.5, 0.5])
    val_loss = Binary_criterion
    train_losslist = []
    epoch = []
    val_ioulist= []
    train_ioulist = []
    val_losslist = []
    min_loss = 0
    all_train = 0
    for i in range(train_epoch):
        loss_number_sum = 0
        model.train()
        for a, train_data in enumerate(train_loader):
            data, label = train_data
            data = data.to(device)
            label = label.to(device)
            opt.zero_grad()
            pred = model(data)
            loss = Binary_criterion(pred, label)
            loss.backward()
            opt.step()
            loss_number = loss.item()
            val_iou = Binary_validation(model, val_loader)
            if val_iou > min_loss:
                min_loss = val_iou
                print("save model")
                torch.save(model, model_path)
            #print('finish')
            writer.add_scalar('Loss/train', loss_number, all_train)
            all_train = all_train + 1
            loss_number_sum = loss_number_sum + loss_number
        scheduler.step()
        train_iou = Binary_validation(model, train_loader)
        train_loss = loss_number_sum/(a+1)
        val_iou = Binary_validation(model, val_loader)
        val_loss_num = validation_loss(model, val_loader,val_loss)
        writer.add_scalar('train_loss/epoch', train_loss, i)
        writer.add_scalar('train_iou/epoch', train_iou, i)
        writer.add_scalar('val_loss/epoch', val_loss_num, i)
        writer.add_scalar('val_iou/epoch', val_iou, i)
        train_losslist.append(train_loss)
        train_ioulist.append(train_iou)
        val_ioulist.append(val_iou)
        val_losslist.append(val_loss_num)
        epoch.append(i)
        print('finish' + str(i) + 'epoch')
    #torch.save(model.state_dict(), last_model)
    return train_losslist, train_ioulist, val_losslist, val_ioulist ,epoch



if __name__ == '__main__':
    star = time.time()
    best_model_path = r''
    last_model = r''
    train_img = r''
    train_label = r''
    txt_Path = r""
    train_loss, train_iou, val_loss, val_iou, epochs =train(train_img, train_label, 'adwas',256, best_model_path, last_model)
    file_write = open(txt_Path, 'w')
    plt.plot(epochs, train_loss, label = 'train loss')
    plt.plot(epochs, val_loss, label = 'val loss')
    plt.plot(epochs, train_iou, label='train acc')
    plt.plot(epochs, val_iou, label='val acc')
    plt.legend()
    plt.savefig("train loss and val mIoU.png", dpi=600)
    plt.show()
    for i in range(100):
        var = str(train_loss[i])+','+str(train_iou[i])+','+str(val_loss[i])+','+str(val_iou[i])
        file_write.write(var)
        file_write.write('\n')
    file_write.close()
