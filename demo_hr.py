from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
import pdb
plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)
import os
import argparse
import sys
sys.path.append("..")

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import ImageDataProvider, ImageDataSingle
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--predict", help="predict only", action="store_true")
args = parser.parse_args()
predict_only = args.predict
print("bool: ", predict_only)

# cmap = {"background":[0,0,0], "lane_line":[128,0,0], "tree":[0,128,0], "tank":[128,128,0], "building":[0,0,128], "plane":[128,0,128], "wzw":[0,128,128]}
label_cmap = {0:[0,0,0], 1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128]}

# output_path = "log_cj_adam"
output_path = "log_l3"
data_provider = ImageDataSingle("./data/cj_cut.png", "./data/cj_cut_gt.png")
# data_provider = ImageDataSingle("./data/cj_test1.png", "./data/cj_test1_gt.png")

class_weights = np.ones(data_provider.n_class)*5
class_weights[0] = 0.5
net = unet.Unet(data_provider.channels, data_provider.n_class, layers=3,
                features_root=32, cost_kwargs={"class_weights": class_weights})

# trainer = unet.Trainer(net, optimizer="momentum", batch_size=10, verification_batch_size=3, opt_kwargs=dict(learning_rate=0.02))
trainer = unet.Trainer(net, optimizer="adam", batch_size=3, verification_batch_size=3, opt_kwargs=dict(learning_rate=0.001))
if not predict_only:
    path = trainer.train(data_provider, output_path, training_iters=40, epochs=500, restore=True)

# path = trainer.train(data_provider, output_path, training_iters=30, epochs=100)

# predict
def stack_imgs(imgs, num_row, num_col):
    '''
    concatenate image slices to a panoroma,
    imgs: image slices should be sorted by row first
    '''
    imgs_row = []
    for i in range(num_row):
        imgs_row.append(np.concatenate(imgs[i*num_col:(i+1)*num_col], axis=1))
    img_stack = np.concatenate(imgs_row, axis=0)
    return img_stack.astype(np.uint8)

path = os.path.join(output_path, "model.ckpt")
images = data_provider.images_origin
masks = data_provider.masks_origin
prediction = net.predict(path, images)

gts = np.argmax(masks, axis=-1)
images_crop = util.crop_to_shape(images, prediction.shape)
gts_crop = util.crop_to_shape(gts, prediction.shape)


label_cmap_list = np.array(list(label_cmap.values()))
preds_rgb = []
preds_gt = []
for i, pred in enumerate(prediction):
    h, w = pred.shape[0:2]
    label = pred.argmax(axis=-1)
    preds_gt.append(label)
    label = label.reshape([-1])
    label_rgb = np.array(label_cmap_list[label])
    label_rgb = label_rgb.reshape((h, w, 3))
    preds_rgb.append(label_rgb)

img = stack_imgs(images_crop, data_provider.num_row, data_provider.num_col)
img_gt = stack_imgs(gts_crop, data_provider.num_row, data_provider.num_col)
pred_rgb = stack_imgs(preds_rgb, data_provider.num_row, data_provider.num_col)
pred_gt = stack_imgs(preds_gt, data_provider.num_row, data_provider.num_col)
cv.imwrite("pred_rgb.png", pred_rgb)
cv.imwrite("img.png", img)
img_add = cv.addWeighted(img, 0.8, pred_rgb, 0.2, 0)
cv.imwrite("img_add.png", img_add)
uniq = img_gt == pred_gt
print("accuracy: %.4f" %(uniq.sum()/np.product(img_gt.shape)))
