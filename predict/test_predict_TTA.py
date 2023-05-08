import torch
from osgeo import gdal
import numpy as np
import glob
import cv2
from attention_deeplab import BNDDeepLab
from image_utils import readTif_info

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

def predict(Model_Path, img_nor,path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BNDDeepLab(in_ch=4, num_classes=1, backbone="resnet34", downsample_factor=16)
    model.to(device)
    model.load_state_dict(torch.load(Model_Path,map_location=torch.device('cpu')))
    #Pred_list = []
    #a = 0
    model.eval()
    for i in range(len(img_nor)):
        with torch.no_grad():
            name = img_nor[i].split('_')[-1]
            im_data,tan, proj = readTif_info(img_nor[i])
            im_data = prepro(im_data)
            im_data = np.array(im_data)
            im_data_hor = np.flip(im_data, axis=2) 
            im_data_vec = np.flip(im_data, axis=1)
            im_data_dia = np.flip(im_data_vec, axis=2)
            im_data_rotz = np.rot90(im_data, -1, (1, 2))
            im_data_rotn = np.rot90(im_data, 1, (1, 2))
            img = np.array([im_data,im_data_hor,im_data_vec,im_data_dia,im_data_rotz,im_data_rotn])
            preds = []
            for j in img:
                img_tensor = torch.from_numpy(np.array([j]))
                img_tensor = img_tensor.to(device=device)
                pred = model(img_tensor)
                if class_style == 'two_class':
                    pred = torch.sigmoid(pred)
                    pred = np.array(pred.data.cpu()[0])[0]
                else:
                    pred = pred[0]
                    pred = torch.softmax(pred, dim=0).cpu()
                    _, pred = torch.max(pred, dim=0)
                    pred = np.array(pred)
                preds.append(pred)
            preds = np.array(preds)
            #print(preds.shape)
            preds[1] = np.flip(preds[1], axis=1)
            preds[2] = np.flip(preds[2], axis=0)
            preds[3] = np.flip(np.flip(preds[3], axis=0), axis=1)
            preds[4] = np.rot90(preds[4], 1, (0, 1))
            preds[5] = np.rot90(preds[5], -1, (0, 1))
            pred = (preds[0]+preds[1]+preds[2]+preds[3]+preds[4]+preds[5])/6
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            #print(pred.shape)
        #Pred_list.append(pred)
        savepath = path + '/result_' + name
        writeTiff(pred, tan, proj, savepath)
        print('finish')


def prepro(data):
    result = np.zeros(data.shape, dtype=np.float32)
    cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
    return result

def normalization_generator(data_list):
    for data in data_list:
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(src=data, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=None)
        yield result

if __name__ == '__main__':
    model_path = r''
    img_list = r""
    class_style = 'two_class'
    savep = r''
    img_list = glob.glob(img_list)
    pred_list = predict(model_path, img_list, savep)
