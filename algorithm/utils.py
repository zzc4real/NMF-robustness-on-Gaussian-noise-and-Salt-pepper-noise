"""
@Author : Huanghao, Shangyue, Zhecheng
@contact: henryzhong4@hotmail.com
@File   : utils.py
@Time   : 5th October, 2020
@Des    : utility function, reference from a#1 example code
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def load_data(root='data/CroppedYaleB', reduce=4):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """
    images, labels = [], []
    image_size = 0

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.

            if reduce:
                # reduce computation complexity.
                image_size = [size // reduce for size in img.size]
                img = img.resize([s // reduce for s in img.size])

            else:
                image_size = img.size
            # TODO: preprocessing.

            # # convert image to numpy array.
            # img = np.asarray(img).reshape((-1, 1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    return images, labels,image_size


def diagram_plot(x, y, x_name, y_name, title):
    """
    :param x: x-axis data
    :param x: x-axis name
    :param y: y-axis data
    :param y: y-axis name
    :param title: image title
    :return: none
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(x, y, color='r', linewidth=3)
    plt.savefig("diagram/{0}.jpg".format(title))
    plt.close()


def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred


def evaluation(H, Y_hat):
    print('==> Evaluate Acc and NMI <==')
    # Assign cluster labels.
    Y_pred = assign_cluster_label(H, Y_hat)

    acc = accuracy_score(Y_hat, Y_pred)
    nmi = normalized_mutual_info_score(Y_hat, Y_pred)
    print('Acc(NMI) = {:.4f} ({:.4f})'.format(acc, nmi))
    return acc, nmi


def write_log(data):
    """
    :param data: write data
    :return:
    """
    with open('result.txt', "a") as file:
        file.write(data)
    file.close()
