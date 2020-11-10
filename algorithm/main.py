"""
@Author : Huanghao, Shangyue, Zhecheng
@contact: henryzhong4@hotmail.com
@File   : main.py
@Time   : 5th October, 2020
@Des    : main function, read image data and execute NMF algorithm (param tuning)
"""

import numpy as np
from utils import load_data
from core import gaussian_test_nmf
from core import salt_pepper_test_nmf

if __name__ == '__main__':
    # read YaleB or ORL data through root
    yale_data, yale_data_label, image_size = load_data(root='data/CroppedYaleB', reduce=4)

    # orl_data, orl_data_label, image_size = load_data(root='data/ORL', reduce=4)
    # print("standardized input image size is: ", image_size)

    # image data is shuffled
    shuffle = True

    # types of noise (execute)
    Gaussian_noise = True
    salt_pepper_noise = True

    # proportion of training data
    train_prop = 0.9

    # salt and pepper noise
    salt_pepper_mode = ['salt_and_pepper', 'salt', 'pepper']

    # proportion of noise
    pol_portion_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

    # num of iteration
    iteration = 2000

    # type of loss function
    loss_type = ['l2', 'l1']

    # type of regularization
    reg_type = ['non']

    if shuffle:
        c = list(zip(yale_data, yale_data_label))
        np.random.shuffle(c)
        x_batch, y_batch = zip(*c)
        yale_data = x_batch
        yale_data_label = y_batch

    # num of input images
    num_input = len(yale_data)
    # print("number of input images: ", num_input)
    # default dimension of dictionary is the num of different image labels
    k_features = len(set(yale_data_label[:num_input]))
    # print("default dimension of dictionary is: ", k_features)
    image_label = np.array(yale_data_label)

    # algorithm tests on Gaussian noise
    if Gaussian_noise:
        for i in loss_type:
            print("**** Start Gaussian Noise {0}_NMF Test ****".format(i))
            gaussian_test_nmf(yale_data, image_label, num_input=num_input, image_size=image_size,
                              k_features=k_features, loss=i, reg=reg_type[0], num_iter=iteration, portion_img=train_prop)
            print('**** Finish Gaussian Noise {0}_NMF Test ****'.format(i))
            print("__________________________________________________")

    # algorithm tests on salt and pepper noise
    if salt_pepper_noise:
        for i in loss_type:
            for m in salt_pepper_mode:
                for j in pol_portion_list:
                    print("**** Start {0}_{1}_{2}_NMF Test ****".format(m, j, i))
                    salt_pepper_test_nmf(yale_data, image_label, num_input=num_input, image_size=image_size,
                                         k_features=k_features, loss=i, reg=reg_type[0], pol_portion=j,
                                         noise_mode=m, num_iter=iteration, portion_img=train_prop)
                    print("**** Finish {0}_{1}_{2}_NMF Test ****".format(m, j, i))
                    print("__________________________________________________")
