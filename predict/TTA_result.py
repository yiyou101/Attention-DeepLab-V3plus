import torch
from osgeo import gdal
import numpy as np
import glob
import math
import segmentation_models_pytorch as smp
import cv2
from GDILDeepLabmodel import GNDDeepLab
from BDILDeepLabmodel import BNDDeepLab

train_length = 1024

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    data = dataset.ReadAsArray(0, 0, width, height)
    #获取地理信息
    geotans = dataset.GetGeoTransform()
    #获取投影
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
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def clipTiff(data, cliplength):
    #存储裁剪数据
    clip_tiff_list = []
    #  列上图像块数目(共几行)
    RowNum = int((data.shape[1] - cliplength * 2) / (train_length - cliplength * 2))
    #  行上图像块数目(共几列)
    ColumnNum = int((data.shape[2] - cliplength * 2) / (train_length - cliplength * 2))
    for i in range(RowNum):
        for j in range(ColumnNum):
            clip = data[:, i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length,
                           j * (train_length - cliplength * 2) : j * (train_length - cliplength * 2) + train_length]
            clip_tiff_list.append(clip)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #如果有剩余的情况往前裁剪一列
    remainder_column = (data.shape[2] - cliplength * 2) % (train_length - cliplength * 2)
    if remainder_column == 0:
        Column_sum = ColumnNum
    else:
        for i in range(RowNum):
            clip = data[:,i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length,
                        (data.shape[2] - train_length) : data.shape[2]]
            clip_tiff_list.insert((i + 1)*ColumnNum + i, clip)
        Column_sum = ColumnNum + 1
    #如果有剩余的情况往前裁剪一行
    remainder_row = (data.shape[1] - cliplength * 2) % (train_length - cliplength * 2)
    if remainder_row == 0:
        Row_sum = RowNum
    else:
        for i in range(ColumnNum):
            clip = data[:, (data.shape[1] - train_length): data.shape[1],
                       i * (train_length - cliplength * 2) : i * (train_length - cliplength * 2) + train_length]
            clip_tiff_list.append(clip)
        Row_sum = RowNum + 1
    #最后右下角一小块空缺
    if remainder_column != 0 and remainder_row != 0:
        clip = data[ : , (data.shape[1] - train_length): data.shape[1], (data.shape[2] - train_length): data.shape[2]]
        clip_tiff_list.append(clip)
    return clip_tiff_list, Row_sum, Column_sum, remainder_row, remainder_column


#影像归一化处理，并且生成生成器
def normalization_generator(data_list):
    for data in data_list:
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
        yield result


#影像拼接
def mosicTiff(pred_data_list, row_sum, col_sum, remainder_row, remainder_column, shape, cliplength):
    #创建一个全是0的数组用来储存拼接的值
    result = np.zeros(shape)
    #共多少行数据
    #样本大小减去裁剪大小的边,左上角影像拼接的边长
    a_length = train_length - cliplength
    # 样本大小减去2个裁剪大小的边,左上角影像拼接的边长
    b_length = train_length - 2*cliplength
    for i,item in enumerate(pred_data_list):
        #裁剪最后都留一行一列，则会在右下角留一小块
        #即适用于有剩余也适用于没有剩余的情况
        #if remainder_row !=0 and  remainder_column != 0:
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if i % col_sum == 0:
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if i == 0:
                result[0 : a_length, 0 : a_length] = item[0 : a_length, 0 : a_length]
            # 最后一行
            elif i/col_sum == row_sum - 1:
                result[shape[0] - remainder_row - cliplength: shape[0], 0 : a_length] = \
                    item[train_length - remainder_row -cliplength: train_length, 0 : a_length]
            else:
                j = int(i/col_sum)
                result[a_length + (j-1) * b_length : a_length + j * b_length, 0 : a_length] = item[cliplength : a_length, 0 :a_length]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i+1) % col_sum == 0:
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if i + 1 == col_sum:
                result[0: a_length, shape[1] - cliplength - remainder_row: shape[1]] = item[0 :a_length, train_length - remainder_row - cliplength: train_length]
            #最后一行，最后一个影像
            elif (i + 1)/col_sum == row_sum :
                result[shape[0] - remainder_row - cliplength: shape[0], shape[1] - remainder_column: shape[1]] = \
                    item[train_length - remainder_row - cliplength: train_length,train_length - remainder_column:train_length]
            else:
                j = int((i + 1) / col_sum) - 1
                result[a_length + (j - 1) * b_length : a_length + j * b_length, shape[1] - cliplength - remainder_column: shape[1]] = \
                    item[cliplength : a_length, train_length - remainder_column - cliplength : train_length]
        else:
            #中间的情况第一行
            if  i > 0 and i < col_sum-1:
                result[0:a_length, a_length + (i-1) *b_length: a_length + i * b_length] = item[0:a_length, cliplength:a_length]
                #中间行
            elif i > col_sum -1 and i < row_sum * col_sum - col_sum:
                #j用来定位第几行，从0开始算起
                j = int(i/col_sum)
                #k用来定位第几列，从0开始算起
                k = i % col_sum
                result[a_length + (j-1) * b_length: a_length + j * b_length, a_length + (k - 1) * b_length: a_length + k * b_length] = \
                    item[ cliplength: a_length, cliplength: a_length]
            else:
                #最后一行的情况
                #j定位最后一行的列数，从0开始算
                j = col_sum - (row_sum * col_sum - i)
                result[shape[0] - remainder_row -cliplength: shape[0], a_length + (j - 1) * b_length: a_length + j * b_length] = \
                    item[train_length - remainder_row - cliplength: train_length, cliplength: a_length]
    return result

#模型语义分割
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
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device)
            pred = model(img_tensor)
            # 提取结果
            if class_style == 'two_class':
                pred = torch.sigmoid(pred)
                #获取onehot类别下标，就可以得出所分出的是哪个类
                print(pred.shape)
                #将四维将为二维
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

#预处理
def processing(im_data,ClipLength, Model_Path):
    # 剪裁影像
    clip_list, row_sum, col_sum, re_row, re_col = clipTiff(im_data, ClipLength)
    print(row_sum, col_sum)
    # 对数据进行归一化处理
    img_normalization = normalization_generator(clip_list)
    # 进行预测
    pred_list = predict(Model_Path, img_normalization)
    result_shape = (im_data.shape[1], im_data.shape[2])
    # 拼接影像
    result = mosicTiff(pred_list, row_sum, col_sum, re_row, re_col, result_shape, ClipLength)
    return result

# 数据增强
def TTA(im_data, ClipLength, Model_Path):
    #  图像水平翻转
    im_data_hor = np.flip(im_data, axis=2)  # 数据是3维
    #  图像垂直翻转
    im_data_vec = np.flip(im_data, axis=1)
    #  图像对角镜像
    im_data_dia = np.flip(im_data_vec, axis=2)
    # 图片顺时针旋转90°
    im_data_rotz = np.rot90(im_data, -1, (1, 2))
    # 图片逆时针旋转90°
    im_data_rotn = np.rot90(im_data, 1, (1, 2))
    #预测
    data_result = processing(img_data, ClipLength, Model_Path)
    data_hor_result = processing(im_data_hor, ClipLength, Model_Path)
    data_vec_result = processing(im_data_vec, ClipLength, Model_Path)
    data_dia_result = processing(im_data_dia, ClipLength, Model_Path)
    data_rotz_result = processing(im_data_rotz, ClipLength, Model_Path)
    data_rotn_result = processing(im_data_rotn, ClipLength, Model_Path)
    #还原
    data_hor_result = np.flip(data_hor_result, axis=1)
    data_vec_result = np.flip(data_vec_result, axis=0)
    data_dia_result = np.flip(np.flip(data_dia_result, axis=0), axis=1)
    data_rotz_result = np.rot90(data_rotz_result, 1, (0, 1))
    data_rotn_result = np.rot90(data_rotn_result, -1, (0, 1))
    tta_result = (data_result + data_hor_result + data_vec_result + data_dia_result + data_rotz_result + data_rotn_result)/6
    return tta_result


model_path = r''
img_list = glob.glob(r"")
class_style = 'two_class'

in_area = 0.8
clipLength = int((1 - math.sqrt(in_area)) * train_length / 2)
img_data, img_trans, img_proj = readTif(img_list[0])
data_result = TTA(img_data , clipLength, model_path)
savepath = r'H:\DL\模型测试\GF2_PMS1_E89.8_N35.6_20201231\att\TTA.tif'
writeTiff(data_result, img_trans, img_proj, savepath)
