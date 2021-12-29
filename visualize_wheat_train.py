# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
from matplotlib import colors

import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

from efficientdet.dataset import WheatDataset

obj_list = ['wheat']
color_list = standard_to_bgr(STANDARD_COLORS)

dset = WheatDataset("datasets/wheat", set="train1")
numSamples = len(dset)
print(f"Got {numSamples} samples")
for idx in range(0, min(numSamples, 10)):
    sample = dset[idx]
    img = sample['img'].copy()
    img = img.astype(np.float32) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(f'test/img_train_{idx}.jpg', img)

    # visualize image & annot
    for annotIdx in range(len(sample['annot'])):
        x1, y1, x2, y2, obj = sample['annot'][annotIdx]
        plot_one_box(img, [x1, y1, x2, y2], label="wheat", score=1.0, color=color_list[0])

    cv2.imwrite(f'test/img_train_{idx}.jpg', img)

