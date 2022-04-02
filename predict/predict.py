import segmentation_models_pytorch as smp
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision
from osgeo import gdal
from torch.utils.data import Dataset,DataLoader
import cv2
from dl_utils import BinaryDiceLoss,openDataset,loss_joint,test_openDataset
from image_utils import writeTiff

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test(train_img, model_path, save_path):
    datalist = test_openDataset(train_img)
    # 定义网络
    print(datalist)
    model = smp.DeepLabV3Plus(encoder_name='resnet34', in_channels=4, classes=1)
    model.to(device)
    # 只用diceloss训练会很暴躁可能不收敛，采用diceloss和ce相结合
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for i, info in enumerate(datalist):
        with torch.no_grad():
            data, trans, proj = info
            img = np.array([data])
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device)
            pred = model(img_tensor)
            # 提取结果
            savepath = save_path + '/' + str(i) + '.tif'
            pred = torch.sigmoid(pred)
            # 获取onehot类别下标，就可以得出所分出的是哪个类
            print(pred.shape)
            # 将四维将为二维
            pred = np.array(pred.data.cpu()[0])[0]
            writeTiff(pred, trans, proj, savepath)
            print('finish'+' '+str(i))

if __name__ == '__main__':
    img_path = r'I:\DL\test_list'
    model_path = 'I:/DL/model/best_model'
    savepath = r'I:\DL\result_list'
    test(img_path, model_path, savepath)