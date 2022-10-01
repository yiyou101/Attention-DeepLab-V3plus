import torch
from osgeo import gdal
import numpy as np
import glob
import math
from BDILDeepLabmodel import BNDDeepLab
import cv2

train_length = 1024

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    geotans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return data, geotans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands,im_height, im_width =im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans) 
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def predict(Model_Path, img_nor):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BNDDeepLab(encoder_name='resnet34', in_channels=4, classes=1)
    model.to(device)
    model.load_state_dict(torch.load(Model_Path))
    Pred_list = []
    #a = 0
    model.eval()
    for i in img_nor:
        with torch.no_grad():
            im_data = np.array(i)
            im_data_hor = np.flip(i, axis=2)  
            im_data_vec = np.flip(i, axis=1)
            im_data_dia = np.flip(im_data_vec, axis=2)
            im_data_rotz = np.rot90(i, -1, (1, 2))
            im_data_rotn = np.rot90(i, 1, (1, 2))
            img = np.array([im_data,im_data_hor,im_data_vec,im_data_dia,im_data_rotz,im_data_rotn])
            print(img.shape)
            preds = []
            for j in img:
                img_tensor = torch.from_numpy(np.array([j]))
                img_tensor = img_tensor.to(device=device)
                pred = model(img_tensor)
                if class_style == 'two_class':
                    pred = torch.sigmoid(pred)
                    print(pred.shape)
                    pred = np.array(pred.data.cpu()[0])[0]
                else:
                    pred = pred[0]
                    pred = torch.softmax(pred, dim=0).cpu()
                    _, pred = torch.max(pred, dim=0)
                    print(pred.shape)
                    pred = np.array(pred)
                preds.append(pred)
                #savepath = 'G:/DL/list2/' + str(a) + '.tif'
                #a = a + 1
                #writeTiff(pred, img_trans, img_proj, savepath)
            preds = np.array(preds)
            print(preds.shape)
            preds[1] = np.flip(preds[1], axis=1)
            preds[2] = np.flip(preds[2], axis=0)
            preds[3] = np.flip(np.flip(preds[3], axis=0), axis=1)
            preds[4] = np.rot90(preds[4], 1, (0, 1))
            preds[5] = np.rot90(preds[5], -1, (0, 1))
            pred = (preds[0]+preds[1]+preds[2]+preds[3]+preds[4]+preds[5])/6
            print(pred.shape)
        Pred_list.append(pred)
        print('finish')
    return Pred_list

def processing(im_data,ClipLength, Model_Path):
    clip_list, row_sum, col_sum, re_row, re_col = clipTiff(im_data, ClipLength)
    print(row_sum, col_sum)
    img_normalization = normalization_generator(clip_list)
    pred_list = predict(Model_Path, img_normalization)
    result_shape = (im_data.shape[1], im_data.shape[2])
    result = mosicTiff(pred_list, row_sum, col_sum, re_row, re_col, result_shape, ClipLength)
    return result

model_path = r''
img_list = glob.glob(r"")
class_style = 'two_class'
in_area = 0.9
clipLength = int((1 - math.sqrt(in_area)) * train_length / 2)
img_data, img_trans, img_proj = readTif(img_list[0])
data_result = processing(img_data , clipLength, model_path)
savepath = r''
writeTiff(data_result, img_trans, img_proj, savepath)
