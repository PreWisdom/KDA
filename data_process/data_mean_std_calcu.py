import os
import cv2
import numpy as np

from tqdm import tqdm


def normalize(imgs_path):
    """
    This function is used to normalize the dataset

    The dataset organization should have the following format:

    ├── dataset
    │   ├── img_dir
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test
    │   ├── ann_dir
    │   │   ├── train
    │   │   ├── val
    │   │   ├── test

    :param imgs_path: img_dir
    :return: norm_mean, norm_std
    """

    img_h, img_w = 512, 512
    means, stdevs = [], []
    img_list, all_img_list = [], []

    for path in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, path)
        img_path_list = os.listdir(img_path)
        for img in img_path_list:
            all_img_list.append(os.path.join(img_path, img))

    i = 0
    for item in tqdm(all_img_list):
        img = cv2.imread(item)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    imgs = imgs.astype(np.float32)
    for i in tqdm(range(3)):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

if __name__ == '__main__':
    img = r'E:\Datasets\CAS_Landslide\Hokkaido\img'
    normalize(img)