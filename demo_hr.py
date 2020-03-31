from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)

import sys
sys.path.append("..")
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import ImageDataProvider, ImageDataSingle

# cmap = {"background":[0,0,0], "lane_line":[128,0,0], "tree":[0,128,0], "tank":[128,128,0], "building":[0,0,128], "plane":[128,0,128], "wzw":[0,128,128]}
label_cmap = {0:[0,0,0], 1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128]}

# data_provider = ImageDataProvider("./data/cj_sep/cj_*.tif")
data_provider = ImageDataSingle("./data/cj_right_all.png", "./data/cj_right_all_mask.png")

net = unet.Unet(channels=3, n_class=7, layers=4, features_root=32)
trainer = unet.Trainer(net, optimizer="momentum", batch_size=50, verification_batch_size=5, opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider, "./test_cj", training_iters=20, epochs=10)
print("path", path)
img = cv.imread("./data/cj_right_all.png", -1)
data = []
nx = 572
for i in range(0, img.shape[1]-nx, nx):
    img_tmp = img[:nx, i:i+nx]
    data.append(img_tmp)
data = np.array(data)

prediction = net.predict(path, data)
data_crop = util.crop_to_shape(data, prediction.shape)
for i, img in enumerate(data_crop):
    pred_img = np.zeros_like(img)
    for l in range(1,7):
        mask = prediction[i, :, :, l] > 0.1
        color = label_cmap[l]
        pred_img[mask] = color
    img_tmp = cv.addWeighted(img, 0.8, pred_img, 0.2, 0)
    cv.imwrite("pred_%d.png" %i, img_tmp)
    cv.imwrite("pred_%d.png" %i, pred_img)

