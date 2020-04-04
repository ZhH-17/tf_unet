from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
import pdb
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

# output_path = "log_cj_adam"
output_path = "log_test_tree_l5_float"
# data_provider = ImageDataSingle("./data/cj_cut.png", "./data/cj_cut_gt.png")
data_provider = ImageDataSingle("./data/cj_test1.png", "./data/cj_test1_gt.png")

class_weights = np.ones(data_provider.n_class)*5
class_weights[0] = 0.5
net = unet.Unet(data_provider.channels, data_provider.n_class, layers=5,
                features_root=32, cost_kwargs={"class_weights": class_weights})
# net = unet.Unet(data_provider.channels, data_provider.n_class, cost="dice_coefficient", layers=3,
#                 features_root=32, cost_kwargs={"class_weights": class_weights})

# trainer = unet.Trainer(net, optimizer="momentum", batch_size=10, verification_batch_size=3, opt_kwargs=dict(learning_rate=0.02))
trainer = unet.Trainer(net, optimizer="adam", batch_size=10, verification_batch_size=3, opt_kwargs=dict(learning_rate=0.002))
path = trainer.train(data_provider, output_path, training_iters=20, epochs=200)
# path = trainer.train(data_provider, output_path, training_iters=30, epochs=100)



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
    for l in range(1,data_provider.n_class):
        mask = prediction[i, :, :, l] > 0.1
        color = label_cmap[l]
        pred_img[mask] = color
    img_tmp = cv.addWeighted(img, 0.8, pred_img, 0.2, 0)
    cv.imwrite("pred_%d.png" %i, img_tmp)
    cv.imwrite("pred_%d.png" %i, pred_img)

