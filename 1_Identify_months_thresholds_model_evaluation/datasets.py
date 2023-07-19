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
from bound_dist import get_boundary,get_distance

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

class Planet_Dataset_Masked(gluon.data.Dataset):
    '''
    masked Planet dataset with crop field extent labels
    '''
    def __init__(self, fold='train', image_names=None,extent_names=None,bound_names=None,random_crop=True):
        self.fold=fold
        self.image_names = image_names
        self.extent_names = extent_names
        self.bound_names=bound_names
        self.random_crop = random_crop # if to randomly crop

    def __getitem__(self, item):
        
        # path of input image and labels
        image_path = self.image_names[item]
        extent_path = self.extent_names[item]
        bound_path=self.bound_names[item]
        
        # read image and label files using imageio
        image=imageio.imread(image_path)
        extent=imageio.imread(extent_path)
        bound=imageio.imread(bound_path)
        
         # get size of the input image
        nrow, ncol, nchannels = image.shape
        
        # augumentation on training dataset
        if self.fold == 'train':
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255)
            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            image = np.rot90(image, k, axes=(0,1))
            extent = np.rot90(extent, k, axes=(0,1))
            bound = np.rot90(bound, k, axes=(0,1))
            # flip augmentation
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=0)
                extent = np.flip(extent, axis=0)
                bound = np.flip(bound, axis=0)
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=1)
                extent = np.flip(extent, axis=1)
                bound = np.flip(bound, axis=1)
#       # convert to uint8
#         image = image.astype(np.uint8) 
        
        # calculate normalised distance
        distance= get_distance(extent)
        
        # define label mask: union of boundary and extent
        label_mask = np.array((extent + bound) >= 1, dtype=np.float32)
        
        # swap dimension for image
        image = mx.nd.array(np.moveaxis(image, -1, 0)) # 256*256*3 -> 3*256*256
        
        # add dimension to all masks: 1*256*256
        extent = mx.nd.array(np.expand_dims(extent, 0))
        bound = mx.nd.array(np.expand_dims(bound, 0))
        distance = mx.nd.array(np.expand_dims(distance, 0))
        label_mask = mx.nd.array(np.expand_dims(label_mask, 0))
        
        return image, extent, bound, distance, label_mask
    
    def __len__(self): # return number of images
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