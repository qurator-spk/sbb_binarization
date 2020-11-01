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
from keras.models import load_model
sys.stderr = stderr
import tensorflow as tf

import logging

def resize_image(img_in, input_height, input_width):
    return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)  # tf.InteractiveSession()

    def end_session(self):
        self.session.close()
        del self.session

    def load_model(self, model_name):
        model = load_model(join(self.model_dir, model_name), compile=False)
        model_height = model.layers[len(model.layers)-1].output_shape[1]
        model_width = model.layers[len(model.layers)-1].output_shape[2]
        n_classes = model.layers[len(model.layers)-1].output_shape[3]
        return model, model_height, model_width, n_classes

    def predict(self, model_in, img, use_patches):
        model, model_height, model_width, n_classes = model_in

        if use_patches:

            margin = int(0.1 * model_width)

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

                    label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

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
