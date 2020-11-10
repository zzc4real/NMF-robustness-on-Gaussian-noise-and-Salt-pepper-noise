"""
@Author : Huanghao, Shangyue, Zhecheng
@contact: henryzhong4@hotmail.com
@File   : algorithm.py
@Time   : 5th October, 2020
@Des    : NMF algorithm implementation
"""

import numpy as np
from utils import diagram_plot


def nmf(x_hat, x, k_features, num_iter, res, noise_type, loss='l2', reg='non', pol_portion=''):
    """
    :param x_hat:       original image and is reshaped
    :param x:           noise + image and is reshaped
    :param k_features:  size and degree of the dictionary
    :param num_iter:    number of iteration default is 1000
    :param res:         the least error that could accept (stopping point value)
    :param noise_type:  type of noise
    :param loss:        type of loss function on NMF
    :param reg:         whether implementing regularization
    :return: d:         dictionary
             r:         weights matrix
    """
    num_pixels, num_images = x.shape
    error_rec, rre_rec, temp_count = [], [], []

    # define the shape of d and r
    d = np.random.random((num_pixels, k_features))
    r = np.random.random((k_features, num_images))

    # L2 norm NMF algorithm without regularization
    if loss == 'l2' and reg == 'non':
        for i in range(num_iter):

            # fix D, solve for R
            r = np.multiply(r, np.divide(np.dot(d.T, x), np.dot(np.dot(d.T, d), r)))

            # fix R, solve for D
            d = np.multiply(d, np.divide(np.dot(x, r.T), np.dot(np.dot(d, r), r.T)))

            # record the training error
            cur_res = np.linalg.norm(x_hat - np.dot(d, r))
            error_rec.append(cur_res)

            # Relative Reconstruction Errors (RRE)
            cur_rre = np.linalg.norm(x_hat - d.dot(r)) / np.linalg.norm(x_hat)
            rre_rec.append(cur_rre)

            temp_count.append(i)

            if i % 100 == 0:
                print('{0}_{1}_L2-NMF_{2}_{3}'.format(noise_type,pol_portion, i, num_iter))
            if i == 0:
                continue
            else:
                if np.abs(error_rec[i - 1] - cur_res) < res:
                    break
        # plot error curve
        diagram_plot(temp_count, error_rec, 'iteration', 'loss', '{0}_{1}_L2-NMF_Loss Function Error'.format(noise_type, pol_portion))
        # plot RRE curve
        diagram_plot(temp_count, rre_rec, 'iteration', 'Error %', '{0}_{1}_L2-NMF_Relative Reconstruct Error'.format(noise_type, pol_portion))


    # L1 norm NMF algorithm without regularization
    elif loss == 'l1' and reg == 'non':
        for i in range(num_iter):
            w = 1 / (np.sqrt(np.square(x - np.dot(d, r))) + 0.0000000001)

            # fix D, solve for R
            r = np.multiply(r, np.dot(d.T, np.multiply(x, w)) / np.dot(d.T, np.multiply(np.dot(d, r), w)))

            # fix R, solve for D
            d = np.multiply(d, np.dot(np.multiply(x, w), r.T) / np.dot(np.multiply(np.dot(d, r), w), r.T))

            # record the training error
            cur_res = np.linalg.norm(x_hat - np.dot(d, r))
            error_rec.append(cur_res)

            # Relative Reconstruction Errors (RRE)
            cur_rre = np.linalg.norm(x_hat - d.dot(r)) / np.linalg.norm(x_hat)
            rre_rec.append(cur_rre)

            temp_count.append(i)

            if i % 100 == 0:
                print('{0}_L1-NMF_{1}_{2}'.format(noise_type, i, num_iter))
            if i == 0:
                continue
            else:
                if np.abs(error_rec[i - 1] - cur_res) < res:
                    break
            # plot error curve
        diagram_plot(temp_count, error_rec, 'iteration', 'loss', '{0}_{1}_L1-NMF_Loss Function Error'.format(noise_type, pol_portion))
        # plot RRE curve
        diagram_plot(temp_count, rre_rec, 'iteration', 'Error %', '{0}_{1}_L1-NMF_Relative Reconstruct Error'.format(noise_type, pol_portion))

    return d, r, rre_rec[-1], error_rec[-1]
