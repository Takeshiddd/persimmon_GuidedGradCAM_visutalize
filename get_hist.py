# coding: utf-8
import os
import sys
EPS = sys.float_info.epsilon
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm 
from PIL import Image
from PIL import ImageEnhance 
import json
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import cv2





def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def make_directry(dir_paths):
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def get_distance_transform(path, save=False, alpha=0.7):
    # カキの陰になっている部分の彩度を向上．彩度変更は PILが便利
    image1 =Image.open(fname)
    con = ImageEnhance.Color(image1)
    con_image = con.enhance(3)

    # PIL型→ OpenCV型
    img = pil2cv(con_image)


    # カラークラスタリング K=2
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # ２値化
    img_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    thr, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # モルフォロジによるノイズ除去 
    kernel1 = np.ones((20,20),np.uint8)
    img_opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel1)
    kernel2 = np.ones((50,50),np.uint8)
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel2)
    img_closing[400,1100:400:800] *= 0

    # ブレンドして表示
    img_new = cv2.cvtColor(img_closing, cv2.COLOR_GRAY2BGR)
    img_dst = (img.astype(np.float) * img_new / 255).astype(np.uint8)
    # img_dst = cv2.addWeighted(img,0.3,img_new,0.7,0)

    # 外部領域からの距離のヒストグラムを取得
    dist_transform = cv2.distanceTransform(img_closing.astype(np.uint8), cv2.DIST_L2, 3)
    
    if save:
        im_name = os.path.basename(fname)
        cv2.imwrite(os.path.join(combined_data_root, im_name), img_dst)
        cv2.imwrite(os.path.join(binary_data_root, im_name), img_new)
        cv2.imwrite(os.path.join(dist_trans_root, im_name), dist_transform / dist_transform.max() * 255)
        

    return dist_transform, img_bin.astype(np.bool)


def get_GuidedGradCAM_weight(data_dict, im_name, reshaped_size=None):
    _map = np.array(data_dict[im_name])
    _map = np.abs(_map).sum(-1)
    if reshaped_size is not None:
        _map = cv2.resize(_map, reshaped_size[::-1])
    _map /= _map.max()
    return _map
    
def make_directory(dir_paths):
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

# path arangement
combined_data_root = 'GGC-pictures_bg_cut'
binary_data_root = 'GGC-pictures_binary'
dist_trans_root = 'GGC-pictures_distanse_transform'
histgram = 'histgram'
hist_data = os.path.join(histgram, 'hist_data')

if __name__ == "__main__":
    make_directory([combined_data_root, binary_data_root, binary_data_root, histgram, hist_data])
    read_file = True  
    dist_normalize = True    # 距離を0-1に正規化する場合True
    conditional = True  # 輪郭からの距離の違いによる画素数不均一を正規化する場合True
    data_name = 'GCC-raw-data' # 読み込むjsonデータの名前を指定
    dist_max = 400
    num_dist_bins = 400
    num_weight_bins = 400
    
    tag = ''
    if dist_normalize:
        tag = '_dist-norm'
        dist_max = 1
    if conditional:
        tag += '_condi'    

    # ディレクトリチェック&作成
    make_directry([binary_data_root, combined_data_root, dist_trans_root, histgram, hist_data])

    if not read_file: # readfile=Falseの場合の処理
        input_data_root = 'GGC-pictures'
        files = glob.glob(os.path.join(input_data_root, '*.JPG'))
        with open(os.path.join('GuidedGradCAM_data', data_name + '.json')) as f:
            data_dict = json.load(f)
        binsx = np.linspace(0, dist_max, num_dist_bins + 1)
        binsy = np.linspace(0, 1, num_weight_bins + 1)
        bins = [binsx, binsy]
        H_0 = list(np.histogram2d([],[], bins=bins))
        Hx_0 = list(np.histogram([], bins=binsx))
        Hy_0 = list(np.histogram([], bins=binsy))
        
        for fname in tqdm(files):
            im_name = os.path.basename(fname)
            dist_trans, binary_mask = get_distance_transform(fname, save=True)
            if dist_normalize and dist_trans.max() != 0:
                dist_trans /= dist_trans.max()

            guidedgradcam = get_GuidedGradCAM_weight(data_dict, im_name, dist_trans.shape)

            x = dist_trans[binary_mask]
            y = guidedgradcam[binary_mask]
            H = np.histogram2d(x, y, bins=bins)
            Hx = np.histogram(x, bins=binsx)
            Hy = np.histogram(y, bins=binsy)
            
            H_0[0] += H[0]
            Hx_0[0] += Hx[0]
            Hy_0[0] += Hy[0]


            # save histgram of each images.
            H_each_im = H_0[0].copy()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if conditional:
                H_each_im /= Hx_0[0][:, np.newaxis]

            # 各画像に対するヒストグラムの画像を生成
            # データによりvmaxとvminを適宜変更してしてヒートマップの濃淡を調整してください．
            im = ax.imshow(H_each_im.T + EPS, interpolation='nearest', origin='lower', \
                                 cmap=cm.jet, norm=LogNorm(), vmax=10**-1, vmin=10**-6)
            ax.set_xlabel('Distanse')
            ax.set_ylabel('Guided Grad CAM Weight')
            ax.set_xticks(np.linspace(0, num_dist_bins, 5))
            ax.set_xticklabels(np.linspace(0, dist_max, 5))
            ax.set_yticks(np.linspace(0, num_weight_bins, 5))
            ax.set_yticklabels(np.linspace(0, 1, 5))
            fig.colorbar(im, ax=ax)
            make_directry([os.path.join(histgram, 'histgram_each_image' + tag)]) # ディレクトリの有無をチェック&なければ作成
            plt.savefig(os.path.join(histgram, 'histgram_each_image' + tag, im_name + '.png'))
            plt.close(fig)


        H_0 = H_0[0]
        Hx_0 = Hx_0[0]
        Hy_0 = Hy_0[0]

        # ヒストグラムの配列データを出力
        np.save(os.path.join(hist_data,  'dist-weight_hist2d' + tag + '.npy'), H_0)
        np.save(os.path.join(hist_data, 'dist_hist' + tag + '.npy'), Hx_0)
        np.save(os.path.join(hist_data, 'weight_hist' + tag + '.npy'), Hy_0)
    
    else: # readfile=Trueの場合の処理
        # ヒストグラムの配列データを読み込み
        H_0 = np.load(os.path.join(hist_data, 'dist-weight_hist2d' + tag + '.npy'))
        Hx_0 = np.load(os.path.join(hist_data, 'dist_hist' + tag + '.npy'))
        Hy_0 = np.load(os.path.join(hist_data, 'weight_hist' + tag + '.npy'))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if conditional:
        H_0 /= Hx_0[:, np.newaxis]

    print('Maximum and minimum histogram values all for images (Please refer to this when setting vmax and vmin.)')
    print('*When the maximum value is 0.7310 and the minimum value is 0.0, the histogram (vmax, vmin) for all images')
    print('should be around (10**-1, 10**-5).')
    print(' - Maximum: {}'.format(H_0.max()))
    print(' - Minimum: {}'.format(H_0.min()))
    # 全画像に対するヒストグラムの画像を生成
    # データによりvmaxとvminを適宜変更してしてヒートマップの濃淡を調整してください．
    im = ax.imshow(H_0.T + EPS, interpolation='nearest', origin='lower', \
                        cmap=cm.jet, norm=LogNorm(), vmax=10**-1, vmin=10**-5)
    ax.set_title('')
    ax.set_xlabel('Distanse')
    ax.set_ylabel('Guided Grad CAM Weight')
    ax.set_xticks(np.linspace(0, num_dist_bins, 5))
    ax.set_xticklabels(np.linspace(0, dist_max, 5))
    ax.set_yticks(np.linspace(0, num_weight_bins, 5))
    ax.set_yticklabels(np.linspace(0, 1, 5))
    fig.colorbar(im, ax=ax)
    make_directry([os.path.join(histgram, 'histgram_all_images')]) # ディレクトリの有無をチェック&なければ作成
    plt.savefig(os.path.join(histgram, 'histgram_all_images', 'dist-weight_hist' + tag + '.png'))
    plt.show()
