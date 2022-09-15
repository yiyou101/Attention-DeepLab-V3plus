from  osgeo import gdal
import numpy as np
import glob
import cv2
from PIL import Image,ImageChops,ImageEnhance

def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")

    width = dataset.RasterXSize

    height = dataset.RasterYSize

    bands = dataset.RasterCount

    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

    geotrans = dataset.GetGeoTransform()

    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


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
        im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans) 
        dataset.SetProjection(im_proj) 
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def img_enhance(img_data):
    random_factor1 = np.random.randint(5, 20) / 10.  
    color_image = ImageEnhance.Color(img_data).enhance(random_factor1)
    random_factor2 = np.random.randint(5, 21) / 10.
    brightness_image = ImageEnhance.Brightness(img_data).enhance(random_factor2)
    random_factor3 = np.random.randint(5, 20) / 10.
    contrast_image = ImageEnhance.Contrast(img_data).enhance(random_factor3)
    random_factor4 = np.random.randint(5, 20) / 10.
    sharp_image = ImageEnhance.Sharpness(img_data).enhance(random_factor4)
    return color_image, brightness_image, contrast_image, sharp_image

def enhance_data(imageList,labelList,train_image_path,train_label_path):
    for i in range(len(imageList)):
        im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(imageList[i])
        imagename = imageList[i][-5:]
        print(imagename)
        im_width, im_height, im_bands, label, im_geotrans, im_proj = readTif(labelList[i])
        im_data_hor = np.flip(im_data, axis=2)  
        hor_path = train_image_path + '/' + 'horizontal_' + imagename
        writeTiff(im_data_hor, im_geotrans, im_proj, hor_path)
        Hor = np.flip(label, axis=1)
        hor_path = train_label_path + '/' + "horizontallabel_" + imagename
        #writeTiff(Hor, im_geotrans, im_proj, hor_path)
        im_data_vec = np.flip(im_data, axis=1)
        vec_path = train_image_path + '/' + 'perpendicular_' + imagename
        writeTiff(im_data_vec, im_geotrans, im_proj, vec_path)
        Vec = np.flip(label, axis=0)
        vec_path = train_label_path + '/' + "perpendicularlaebl_" + imagename
        #writeTiff(Vec, im_geotrans, im_proj, vec_path)
        im_data_dia = np.flip(im_data_vec, axis=2)
        dia_path = train_image_path + '/' +'diagonally_' + imagename
        writeTiff(im_data_dia, im_geotrans, im_proj, dia_path)
        Dia = np.flip(Vec, axis=1)
        dia_path = train_label_path + '/' + "diagonallylabel_" + imagename
        #writeTiff(Dia, im_geotrans, im_proj, dia_path)
        im_data_rotz = np.rot90(im_data, -1, (1,2))
        rotz_path = train_image_path + '/' + 'rotz_' + imagename
        writeTiff(im_data_rotz, im_geotrans, im_proj, rotz_path)
        rotz = np.rot90(label, -1)
        rotz_path = train_label_path + '/' +'rotzlabel_' + imagename
        #writeTiff(rotz, im_geotrans, im_proj, rotz_path)
        im_data_rotn = np.rot90(im_data, 1, (1,2))
        rotn_path = train_image_path + '/' +'rotn_' + imagename
        writeTiff(im_data_rotn, im_geotrans, im_proj, rotn_path)
        rotn = np.rot90(label, 1)
        rotn_path = train_label_path + '/' + 'rotnlabel_' + imagename
        #writeTiff(rotn, im_geotrans, im_proj, rotn_path)
        color_img, brightness_img, contrast_img, sharp_img = img_enhance(im_data)
        color_path = train_image_path + '/' + 'color' + imagename
        color_label = train_image_path + '/' + 'colorlabel_' + imagename
        writeTiff(color_img, im_geotrans, im_proj, color_path)
        #writeTiff(label, im_geotrans, im_proj, color_label)
        brightness_path = train_image_path + '/' + 'brightness' + imagename
        brightness_label = train_image_path + '/' + 'brightnesslabel_' + imagename
        writeTiff(brightness_img, im_geotrans, im_proj, brightness_path)
        #writeTiff(label, im_geotrans, im_proj, brightness_label)
        contrast_path = train_image_path + '/' + 'contrast' + imagename
        contrast_label = train_image_path + '/' + 'contrastlabel_' + imagename
        writeTiff(contrast_img, im_geotrans, im_proj, contrast_path)
        #writeTiff(label, im_geotrans, im_proj, contrast_label)
        sharp_path = train_image_path + '/' + 'sharp' + imagename
        sharp_label = train_image_path + '/' + 'sharplabel_' + imagename
        writeTiff(sharp_img, im_geotrans, im_proj, sharp_path)
        #writeTiff(label, im_geotrans, im_proj, sharp_label)

if __name__ == '__main__':
    train_image_path = r""
    train_label_path = r""
    imageList = glob.glob(r"")
    labelList = glob.glob(r"")
    print(imageList)
    print(labelList)
    enhance_data(imageList,labelList,train_image_path,train_label_path)
