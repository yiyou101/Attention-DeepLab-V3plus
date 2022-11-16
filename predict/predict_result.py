import torch
from osgeo import gdal
import numpy as np
import glob
import math
import cv2
from Attention_DeepLab import BNDDeepLab
from deeplab import deeplab

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

def clipTiff(data, cliplength):
    clip_tiff_list = []
    RowNum = int((data.shape[1] - cliplength * 2) / (train_length - cliplength * 2))
    ColumnNum = int((data.shape[2] - cliplength * 2) / (train_length - cliplength * 2))
    for i in range(RowNum):
        for j in range(ColumnNum):
            clip = data[:, i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length,
                           j * (train_length - cliplength * 2) : j * (train_length - cliplength * 2) + train_length]
            clip_tiff_list.append(clip)
    remainder_column = (data.shape[2] - cliplength * 2) % (train_length - cliplength * 2)
    if remainder_column == 0:
        Column_sum = ColumnNum
    else:
        for i in range(RowNum):
            clip = data[:,i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length,
                        (data.shape[2] - train_length) : data.shape[2]]
            clip_tiff_list.insert((i + 1)*ColumnNum + i, clip)
        Column_sum = ColumnNum + 1
    remainder_row = (data.shape[1] - cliplength * 2) % (train_length - cliplength * 2)
    if remainder_row == 0:
        Row_sum = RowNum
    else:
        for i in range(ColumnNum):
            clip = data[:, (data.shape[1] - train_length): data.shape[1],
                       i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length]
            clip_tiff_list.append(clip)
        Row_sum = RowNum + 1
    if remainder_column != 0 and remainder_row != 0:
        clip = data[ : , (data.shape[1] - train_length): data.shape[1], (data.shape[2] - train_length): data.shape[2]]
        clip_tiff_list.append(clip)
    return clip_tiff_list, Row_sum, Column_sum, remainder_row, remainder_column

def normalization_generator(data_list):
    for data in data_list:
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
        yield result



def mosicTiff(pred_data_list, row_sum, col_sum, remainder_row, remainder_column, shape, cliplength):

    result = np.zeros(shape)

    a_length = train_length - cliplength

    b_length = train_length - 2*cliplength
    for i,item in enumerate(pred_data_list):

        #if remainder_row !=0 and  remainder_column != 0:

        if i % col_sum == 0:

            if i == 0:
                result[0 : a_length, 0 : a_length] = item[0 : a_length, 0 : a_length]

            elif i/col_sum == row_sum - 1:
                result[shape[0] - remainder_row - cliplength: shape[0], 0 : a_length] = \
                    item[train_length - remainder_row -cliplength: train_length, 0 : a_length]
            else:
                j = int(i/col_sum)
                result[a_length + (j-1) * b_length : a_length + j * b_length, 0 : a_length] = item[cliplength : a_length, 0 :a_length]

        elif (i+1) % col_sum == 0:

            if i + 1 == col_sum:
                result[0: a_length, shape[1] - cliplength - remainder_row: shape[1]] = item[0 :a_length, train_length - remainder_row - cliplength: train_length]

            elif (i + 1)/col_sum == row_sum :
                result[shape[0] - remainder_row - cliplength: shape[0], shape[1] - remainder_column: shape[1]] = \
                    item[train_length - remainder_row - cliplength: train_length,train_length - remainder_column:train_length]
            else:
                j = int((i + 1) / col_sum) - 1
                result[a_length + (j - 1) * b_length : a_length + j * b_length, shape[1] - cliplength - remainder_column: shape[1]] = \
                    item[cliplength : a_length, train_length - remainder_column - cliplength : train_length]
        else:

            if  i > 0 and i < col_sum-1:
                result[0:a_length, a_length + (i-1) *b_length: a_length + i * b_length] = item[0:a_length, cliplength:a_length]

            elif i > col_sum -1 and i < row_sum * col_sum - col_sum:

                j = int(i/col_sum)

                k = i % col_sum
                result[a_length + (j-1) * b_length: a_length + j * b_length, a_length + (k - 1) * b_length: a_length + k * b_length] = \
                    item[ cliplength: a_length, cliplength: a_length]
            else:

                j = col_sum - (row_sum * col_sum - i)
                result[shape[0] - remainder_row -cliplength: shape[0], a_length + (j - 1) * b_length: a_length + j * b_length] = \
                    item[train_length - remainder_row - cliplength: train_length, cliplength: a_length]
    return result


def predict(Model_Path, img_nor):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BNDDeepLab(in_ch=4, num_classes=1, backbone="resnet34", downsample_factor=16)
    model.to(device)
    model.load_state_dict(torch.load(Model_Path))
    Pred_list = []
    a = 0
    model.eval()
    for i in img_nor:
        with torch.no_grad():
            img = np.array([i])
            img_tensor = torch.from_numpy(img)
。
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
            #savepath = 'G:/DL/list2/' + str(a) + '.tif'
            a = a + 1
            #writeTiff(pred, img_trans, img_proj, savepath)
            print('finish')
            print(pred.shape)
            Pred_list.append(pred)
            print('finish')
    return Pred_list

if __name__ == '__main__':
    model_path = r''
    img_list = glob.glob(r"")
    class_style = 'two_class'

    in_area = 0.9
    clipLength = int((1 - math.sqrt(in_area)) * train_length / 2)
    img_data, img_trans, img_proj = readTif(img_list[0])

    clip_list, row_sum, col_sum,re_row, re_col = clipTiff(img_data, clipLength)
    print(row_sum,col_sum)

    img_normalization = normalization_generator(clip_list)

    pred_list = predict(model_path,img_normalization)

    result_shape = (img_data.shape[1], img_data.shape[2])
    mosic_result = mosicTiff(pred_list, row_sum, col_sum, re_row, re_col, result_shape, clipLength)
    savepath = r''
    writeTiff(mosic_result, img_trans, img_proj, savepath)
