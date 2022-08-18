"""
Tool to load model and binarize a given image.
"""

import sys
from glob import glob
from os import environ, devnull
from os.path import join
from warnings import catch_warnings, simplefilter

import numpy as np
from PIL import Image
import cv2
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(devnull, 'w')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as tensorflow_backend
from tensorflow.keras import layers
import tensorflow.keras.losses
from tensorflow.keras.layers import *
sys.stderr = stderr


import logging


projection_dim = 64
patch_size = 1
num_patches =14*14

def resize_image(img_in, input_height, input_width):
    return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)


class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    
class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

class SbbBinarizer:

    def __init__(self, model_dir, logger=None):
        self.model_dir = model_dir
        self.log = logger if logger else logging.getLogger('SbbBinarizer')

        self.start_new_session()

        self.model_files = glob('%s/*.h5' % self.model_dir)

        self.models = []
        for model_file in self.model_files:
            self.models.append(self.load_model(model_file))

    def start_new_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
        tensorflow_backend.set_session(self.session)

    def end_session(self):
        tensorflow_backend.clear_session()
        self.session.close()
        del self.session

    def load_model(self, model_name):
        try:
            model = load_model(join(self.model_dir, model_name), compile=False)
            self.margin_percent = 0.1
        except:
            model = load_model(join(self.model_dir, model_name) , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})
            self.margin_percent = 0.15
        model_height = model.layers[len(model.layers)-1].output_shape[1]
        model_width = model.layers[len(model.layers)-1].output_shape[2]
        n_classes = model.layers[len(model.layers)-1].output_shape[3]
        return model, model_height, model_width, n_classes

    def predict(self, model_in, img, use_patches):
        tensorflow_backend.set_session(self.session)
        model, model_height, model_width, n_classes = model_in
        
        img_org_h = img.shape[0]
        img_org_w = img.shape[1]
        
        if img.shape[0] < model_height and img.shape[1] >= model_width:
            img_padded = np.zeros(( model_height, img.shape[1], img.shape[2] ))
            
            index_start_h =  int( abs( img.shape[0] - model_height) /2.)
            index_start_w = 0
            
            img_padded [ index_start_h: index_start_h+img.shape[0], :, : ] = img[:,:,:]
            
        elif img.shape[0] >= model_height and img.shape[1] < model_width:
            img_padded = np.zeros(( img.shape[0], model_width, img.shape[2] ))
            
            index_start_h =  0 
            index_start_w = int( abs( img.shape[1] - model_width) /2.)
            
            img_padded [ :, index_start_w: index_start_w+img.shape[1], : ] = img[:,:,:]
            
            
        elif img.shape[0] < model_height and img.shape[1] < model_width:
            img_padded = np.zeros(( model_height, model_width, img.shape[2] ))
            
            index_start_h =  int( abs( img.shape[0] - model_height) /2.)
            index_start_w = int( abs( img.shape[1] - model_width) /2.)
            
            img_padded [ index_start_h: index_start_h+img.shape[0], index_start_w: index_start_w+img.shape[1], : ] = img[:,:,:]
            
        else:
            index_start_h = 0
            index_start_w  = 0
            img_padded = np.copy(img)
            
        img_org_h_pad = img_padded.shape[0]
        img_org_w_pad = img_padded.shape[1]
            
        index_start_h_alw = 100
        index_start_w_alw = 100
        
        img_padded_alw = np.zeros(( img_padded.shape[0]+2*index_start_h_alw, img.shape[1]+2*index_start_w_alw, img.shape[2] ))
        
        
        img_padded_alw [ 0: index_start_h_alw, index_start_w_alw: index_start_w_alw+img_padded.shape[1], : ] = img_padded[:index_start_h_alw,:,:]
        img_padded_alw [ index_start_h_alw: index_start_h_alw+img_padded.shape[0], 0:index_start_w_alw, : ] = img_padded[:,0:index_start_w_alw,:]
        
        img_padded_alw [ img_padded_alw.shape[0]-index_start_h_alw: img_padded_alw.shape[0], index_start_w_alw: index_start_w_alw+img_padded.shape[1], : ] = img_padded[img_padded.shape[0]-index_start_h_alw:img_padded.shape[0],:,:]
        img_padded_alw [ index_start_h_alw: index_start_h_alw+img_padded.shape[0],img_padded_alw.shape[1]-index_start_w_alw: img_padded_alw.shape[1], : ] = img_padded[:,img_padded.shape[1]-index_start_w_alw:img_padded.shape[1],:]
            
        img_padded_alw [ index_start_h_alw: index_start_h_alw+img_padded.shape[0], index_start_w_alw: index_start_w_alw+img_padded.shape[1], : ] = img_padded[:,:,:]
        
        img = np.copy(img_padded_alw)

        
            

        if use_patches:

            margin = int(self.margin_percent * model_width)

            width_mid = model_width - 2 * margin
            height_mid = model_height - 2 * margin


            img = img / float(255.0)

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - model_width
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - model_height

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    
                    #h_res = int( img_patch.shape[0]/1.05)
                    #w_res = int( img_patch.shape[1]/1.05)
                    
                    #img_patch_resize = resize_image(img_patch, h_res, w_res)
                    
                    #img_patch_resized_padded =np.ones((img_patch.shape[0],img_patch.shape[1],img_patch.shape[2])).astype(float)#self.do_padding()
                    
                    #h_start=int( abs(img_patch.shape[0]-img_patch_resize.shape[0])/2. )
                    
                    #w_start=int( abs(img_patch.shape[1]-img_patch_resize.shape[1])/2. )
                    
                    #img_patch_resized_padded[h_start:h_start+img_patch_resize.shape[0],w_start:w_start+img_patch_resize.shape[1],:]=np.copy(img_patch_resize[:,:,:])
                    
                    #label_p_pred_padded = model.predict(img_patch_resized_padded.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    #seg = np.argmax(label_p_pred, axis=3)[0]

                    #label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    #seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i == 0 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin, :] = seg_color

                    elif i == nxf-1 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0, :] = seg_color

                    elif i == 0 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin, :] = seg_color

                    elif i == nxf-1 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0, :] = seg_color

                    elif i == 0 and j != 0 and j != nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin, :] = seg_color

                    elif i == nxf-1 and j != 0 and j != nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0, :] = seg_color

                    elif i != 0 and i != nxf-1 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin, :] = seg_color

                    elif i != 0 and i != nxf-1 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin, :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin, :] = seg_color
            
            
            prediction_true = prediction_true[index_start_h_alw: index_start_h_alw+img_org_h_pad, index_start_w_alw: index_start_w_alw+img_org_w_pad,:]
            prediction_true = prediction_true[index_start_h: index_start_h+img_org_h, index_start_w: index_start_w+img_org_w,:]
            prediction_true = prediction_true.astype(np.uint8)

        else:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, model_height, model_width)

            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true[:,:,0]

    def run(self, image=None, image_path=None, save=None, use_patches=False):
        if (image is not None and image_path is not None) or \
               (image is None and image_path is None):
            raise ValueError("Must pass either a opencv2 image or an image_path")
        if image_path is not None:
            image = cv2.imread(image_path)
        img_last = 0
        for n, (model, model_file) in enumerate(zip(self.models, self.model_files)):
            self.log.info('Predicting with model %s [%s/%s]' % (model_file, n + 1, len(self.model_files)))

            res = self.predict(model, image, use_patches)

            img_fin = np.zeros((res.shape[0], res.shape[1], 3))
            res[:, :][res[:, :] == 0] = 2
            res = res - 1
            res = res * 255
            img_fin[:, :, 0] = res
            img_fin[:, :, 1] = res
            img_fin[:, :, 2] = res

            img_fin = img_fin.astype(np.uint8)
            img_fin = (res[:, :] == 0) * 255
            img_last = img_last + img_fin

        kernel = np.ones((5, 5), np.uint8)
        img_last[:, :][img_last[:, :] > 0] = 255
        img_last = (img_last[:, :] == 0) * 255
        if save:
            cv2.imwrite(save, img_last)
        return img_last
