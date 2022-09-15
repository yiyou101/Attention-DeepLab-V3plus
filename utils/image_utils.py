import glob
from osgeo import gdal, osr, ogr
import numpy as np
from sklearn import model_selection


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    return data

def readTif_info(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    geotans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return data, geotans, proj

def readTif_moreinfo(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
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
    # elif 'int32' in im_data.dtype.name:
    #     datatype = gdal.GDT_Int32
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
        #srs = osr.SpatialReference()
        #srs.ImportFromEPSG(4326)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def random_split_train_val(image_paths, label_paths, val_index=0):
    image_paths = glob.glob(image_paths + '/*.tif')
    label_paths = glob.glob(label_paths + '/*.tif')
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = model_selection.train_test_split(image_paths ,label_paths, random_state=1, train_size=val_index,test_size=1-val_index)
    print("Number of train images after upsample: ", len(train_image_paths))
    print("Number of val images after upsample: ", len(val_image_paths))
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths
