import numpy as np
# import pandas as pd
import os
# import imageio
import imageio.v2 as imageio
import cv2

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import image
from osgeo import gdal, osr
    
# class PlanetDatasetNoLabels_Lavender(gluon.data.Dataset):
    
#     def __init__(self, image_names):
#         self.image_names = image_names
        
#     def __getitem__(self, item):

#         # read image file using imageio
#         image_path=self.image_names[item]
#         image=imageio.imread(image_path)

#         # scale from surface reflectance (0~10000) to uint8 (0~255)
#         image=(image/10000.0*255).astype(np.uint8)

#         # swap dimension
#         image = mx.nd.array(np.moveaxis(image, -1, 0))

#         return image
    
#     def __len__(self):
#         return len(self.image_names)
    
class Planet_Dataset_No_labels(gluon.data.Dataset):
    '''Planet Dataset in Rwanda'''
    def __init__(self, image_names):
        self.image_names = image_names
        
    def read_img(self,img_path):
        # read image file using imageio
        image=imageio.imread(img_path)
        # swap dimension
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        return image
    
    def extract_img_geotrans(self,img_path):
        '''return geotransform of geotiff as narray'''
        ds=gdal.Open(img_path)
        geotrans=ds.GetGeoTransform()
        ds=None
        return mx.nd.array(geotrans)
    
    def extract_id_date(self,img_path):
        '''return image chunk id and acquisition data information as ndarray
        this function is very customised and depending on your file naming convention'''
        img_name=os.path.basename(img_path).replace('.','_')
        name_elements=img_name.split('_')
        id_date=[int(name_elements[i]) for i in [-5,-4,-3,-2]]
    #     chunk_id_row=int(name_elements[-3])
    #     chunk_id_col=int(name_elements[-2])
    #     year=int(name_elements[-5])
    #     month=int(name_elements[-4])
    #     return mx.nd.array([chunk_id,year,month])
        return mx.nd.array(id_date)
    
    def __getitem__(self, item):
        img_path=self.image_names[item]
        return self.read_img(img_path),self.extract_id_date(img_path),self.extract_img_geotrans(img_path)
    
    def __len__(self):
        return len(self.image_names)


def export_geotiff(outname,arr,geotrans,proj,datatype):
    '''export array to geotiff'''
    rows,cols=arr.shape[0],arr.shape[1]
    if os.path.exists(outname):
        os.remove(outname)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outname, cols, rows, 1, datatype)
    outdata.SetGeoTransform(geotrans)
    outdata.SetProjection(proj)
    outdata.GetRasterBand(1).WriteArray(arr)
    outdata.FlushCache()
    outdata = None