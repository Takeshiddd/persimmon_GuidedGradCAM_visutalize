# coding: utf-8
import numpy as np 
import json
import matplotlib
from matplotlib import pyplot as plt
import os
import cv2
from tqdm import tqdm


def get_guidedgradcam_to_rgb_with_image(guidedgradcam, reshaped_size=None, cmap='seismic'):
    guidedgradcam = guidedgradcam.sum(-1)
    M = np.abs(guidedgradcam).max()
    if M == 0: 
        M = 1
    guidedgradcam += M
    guidedgradcam /= 2 * M
    heat_map = toHeatmap(guidedgradcam[0], cmap)
    if reshaped_size is not None:
        heat_map = cv2.resize(heat_map, reshaped_size[::-1])
    heat_map *= 255
    return heat_map.astype(np.int)

def get_guidedgradcam_to_rgb_with_image_abs(guidedgradcam, reshaped_size=None, cmap='jet'):
    guidedgradcam = guidedgradcam.sum(-1)
    guidedgradcam = np.abs(guidedgradcam)
    guidedgradcam -= guidedgradcam.min()
    M = np.abs(guidedgradcam).max()
    if M == 0: 
        M = 1
    guidedgradcam /= M
    heat_map = toHeatmap(guidedgradcam[0], cmap)
    if reshaped_size is not None:
        heat_map = cv2.resize(heat_map, reshaped_size[::-1])
    heat_map *= 255
    return heat_map.astype(np.int)

def get_guidedgradcam_feature(_map, image_size):
    M = np.abs(_map).max()
    if M == 0:
        M = 1
    _map += M
    _map /= 2 * M
    _map *= 255
    _map = cv2.resize(_map, image_size[::-1])
    return _map

def toHeatmap(x, cmap='seismic'):
    h, w = x.shape
    x = (x*255).reshape(-1)
    cm = plt.get_cmap(cmap)
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return x.reshape(h, w, 3)

def make_directry(dir_paths):
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

if __name__ == "__main__":
    abs_weight = False # Trueの場合，青が0，赤が最大の画像，Falseの場合青がマイナス，赤がプラスの画像
    alpha = 0.5 # 大きいほどヒートマップが濃くなる
    data_name = 'GCC-raw-data' # 読み込むjsonデータの名前を指定

    # ディレクトリチェック&作成
    vis_path = 'GuidedGradCAM_visualization'
    vis_with_image_path = 'GuidedGradCAM_visualization_with_image'
    make_directry([vis_path, vis_with_image_path])

    if abs_weight:
        get_feature_with_image = get_guidedgradcam_to_rgb_with_image_abs
    else:
        get_feature_with_image = get_guidedgradcam_to_rgb_with_image

    with open(os.path.join('GuidedGradCAM_data', data_name + '.json')) as f:
        data_dict = json.load(f)


    cmap = matplotlib.cm.seismic
    for im_name in tqdm(data_dict):
        image = plt.imread(os.path.join('GGC-pictures', im_name))
        guidedgradcam = np.array(data_dict[im_name])
        guidedgradcam_with_im = get_feature_with_image(guidedgradcam[np.newaxis], reshaped_size=image.shape[:2])
        ##################################### GuidedGradCAM_visualization #####################################
        # guidedgradcam_feature = get_guidedgradcam_feature(guidedgradcam, image_size=image.shape[:2])
        # plt.imsave(os.path.join('GuidedGradCAM_visualization', im_name), guidedgradcam_feature.astype(np.uint8))
        
        ################################# GuidedGradCAM_visualization_seismic #################################
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
        ax2 = fig.add_axes([0.8, 0.195, 0.03, 0.61])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
        ax.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
        ax.imshow(guidedgradcam_with_im.astype(np.uint8))
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical')
        fig.savefig(os.path.join('GuidedGradCAM_visualization_seismic', im_name))

        ############################### GuidedGradCAM_visualization_with_image ###############################
        guidedgradcam_with_im = image  * (1 - alpha) + guidedgradcam_with_im * alpha
        plt.imsave(os.path.join('GuidedGradCAM_visualization_with_image', im_name), guidedgradcam_with_im.astype(np.uint8))
        



        
        


        
        


