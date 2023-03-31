# import the necessary packages
import apriltag
import argparse
import cv2
import os
from glob import glob
import numpy as np
import trimesh
import torch
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.cluster import MeanShift
import tqdm

target = [[127, 97, 1],
    [161, 2, 148],
    [249, 241, 72],
    [131, 241, 242],
    [253, 84, 63],
    [104, 68, 252],
    [143, 247, 49]]

src = [[95, 68, 1],
    [117, 0, 105],
    [213, 209, 0],
    [80, 218, 220],
    [221, 52, 47],
    [65, 43, 206],
    [29, 195, 0]]

if __name__ == '__main__':
    
    dir = '../real_data_cat_2/env_matting/'
    src = np.array(src)[:, ::-1]
    target = np.array(target)[:, ::-1]
    for i in range(329, 389):
        print(i)
        image = cv2.imread(dir + str(i) + '.jpg')
        # image = cv2.bilateralFilter(image, 5, 30, 30)
        image = cv2.medianBlur(image, 5)
        image = cv2.medianBlur(image, 5)
        image_pixels = image.reshape((-1, 3))

        K = 1
        kdt = KDTree(src, leaf_size=40, metric='euclidean')
        dist, nn_idx = kdt.query(image_pixels, k=K, return_distance=True)
        dist = dist[:, 0]
        nn_idx = nn_idx[:, 0]
        image_pixels = target[nn_idx, :] 
        # print(target[nn_idx, :])
        # exit()

        # for color_idx in range(7):
            # src_color = src[color_idx, :]
            # target_color = target[color_idx, :]
        
            # diff = image - src_color[np.newaxis, np.newaxis, :] 
            # diff = diff.reshape((-1, 3))
            # diff = np.linalg.norm(diff, axis=1)
            # mask = (diff < 30)
            # image_pixels[mask, :] = target_color 
        image = image_pixels.reshape(image.shape)
        cv2.imwrite(dir + str(i) + '.jpg', image)
        
    # image = image.astype(np.float32) / 255.0