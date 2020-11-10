"""
@Author : Huanghao, Shangyue, Zhecheng
@contact: henryzhong4@hotmail.com
@File   : core.py
@Time   : 5th October, 2020
@Des    : The function is used to perform noise addition and NMF algorithm
"""
import numpy as np
import random
from PIL import Image
# from utils import evaluation_metrics
from algorithm import nmf
from utils import evaluation
from utils import write_log
import os
import sys
import datetime
import matplotlib.pyplot as plt


def add_salt_pepper(images, mode = 'salt_and_pepper', pol_portion = 0.1, portion_img = 1):
    """
     :param images       input image
     :param mode         noise mode
     :param pol_portion  proportion of polluted images
     :param portion_img  proportion of training images
     :return image with salt and pepper noise (type: array)
    """
    num_img = int(len(images) * portion_img)
    row, col = images[0].size
    salt_pepper_images = []

    # add salt and pepper noise to pol_portion of image
    for image in images[:num_img]:
        img = np.array(image)

        # salt noise
        if mode == 'salt':
            mask = np.random.choice(('o', 's'), size=(col, row), p = [1 - pol_portion, pol_portion])
            img[mask == 's'] = 255

        # pepper noise
        elif mode == 'pepper':
            mask = np.random.choice(('o', 'p'), size=(col, row), p = [1 - pol_portion, pol_portion])
            img[mask == 'p'] = 0

        # salt and pepper noise
        elif mode =='salt_and_pepper':
            mask = np.random.choice(('o', 's', 'p'), size=(col, row), p=[1 - pol_portion, pol_portion/2, pol_portion/2])
            img[mask == 's'] = 255
            img[mask == 'p'] = 0

        salt_pepper_images.append(img)

    # add remaining images
    if portion_img < 1:
        for others in images[num_img:]:
            salt_pepper_images.append(np.array(others))

    return salt_pepper_images



def salt_pepper_test_nmf(x_hat, y_hat, num_input, image_size, k_features, loss='l2', reg='non',
                         pol_portion=0.01, portion_img=1, noise_mode='salt_and_pepper', num_iter=2000):
    """
        :param x_hat        input image
        :param y_hat        image label
        :param num_input    a number of total input images
        :param k_features   equals to the number of groups image
        :param noise_mode   salt_and_pepper & salt & pepper
        :param pol_portion  proportion of polluted images
        :param portion_img  proportion of training image
        :param num_iter     number of iteration
    """
    mkdir('diagram')
    salt_pepper_image_path = mkdir('salt_pepper/salt_pepper_img/{0}/{1}/{2}'.format(loss, noise_mode, pol_portion))
    salt_pepper_recon_path = mkdir('salt_pepper/salt_pepper_recon_img/{0}/{1}/{2}'.format(loss, noise_mode, pol_portion))
    salt_pepper_feature_path = mkdir('salt_pepper/salt_pepper_feature/{0}/{1}/{2}'.format(loss, noise_mode, pol_portion))

    # add salt and pepper noise to images
    print("- adding salt and pepper noise to images")
    images = add_salt_pepper(x_hat[:num_input], mode=noise_mode, pol_portion=pol_portion, portion_img=portion_img)

    # save salt and pepper image to salt_pepper_img folder
    count = 1
    for j in images:
        salt_pepper_img = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(salt_pepper_image_path, count), salt_pepper_img, cmap='gray')
        count += 1
    print("images with salt and pepper noise are saved to {0} folder".format(salt_pepper_image_path))

    # reshape salt and pepper image array into a (2016,2414) matrix -> (img.row * img.col, num of img)
    # each col represents an image
    reshape_salt_and_pepper_img = img_array_reshape(images)
    x_hat = img_array_reshape(x_hat)

    # execute NMF algorithm on salt and pepper images
    print("- starting NMF algorithm on {0}_{1} images".format(noise_mode, pol_portion))
    d, r, rre, error = nmf(x_hat=x_hat[:, :num_input], x=reshape_salt_and_pepper_img, noise_type= noise_mode,
                           k_features=k_features, num_iter=num_iter, res=0.0001, loss=loss, reg=reg, pol_portion=pol_portion)

    print("- reconstructing image")
    # reconstruct image
    recon_img = np.dot(d, r)
    count = 1
    for j in recon_img.T:
        pic_1 = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(salt_pepper_recon_path, count), pic_1, cmap='gray')
        count += 1

    print("- evaluating algorithm accuracy")
    # evaluate the accuracy
    acc, nmi = evaluation(r.T, y_hat[:num_input])

    print("- storing dictionary (features) into gaussian_feature directory")
    # visualize the dictionary (features)
    count = 1
    for j in (d.T):
        pic_1 = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(salt_pepper_feature_path, count), pic_1, cmap='gray')
        count += 1

    # save acc,nmi, rre to result.txt
    # [noise type]  [prop]  [loss]  [reg]
    # [loss error]  [rre]   [acc]   [nmi]
    content = "{0} - {1} - loss type:{2} - regularization:{3}\n" \
              "Loss error\t\tRRE\t\t\tACC\t\t\tNMI\n" \
              "{4}\t{5}\t{6}\t{7}\n\n".format(noise_mode, pol_portion, loss, reg, error, rre, acc, nmi)
    write_log(content)



def add_gaussian_noise(images, portion_img, mean=0, var=0.001):
    """
    :parameter images:      input image
    :parameter mean var:    mean & variance of gaussian
    :parameter portion_img: proportion of input images with gaussian noise
    :return:                image
    """
    gaussian_image = []
    port_num = int(len(images) * portion_img)
    temp = 0
    std_var = var ** 0.5

    for image in images:
        if temp < port_num:
            # image is a [0,1] float array
            image = np.array(np.asarray(image) / 255, dtype=float)
            noise = np.random.normal(mean, std_var, image.shape)
            out = image + noise
            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            out = np.clip(out, low_clip, 1.0)
            out = np.uint8(out * 255)
            gaussian_image.append(out)
            temp += 1
        else:
            gaussian_image.append(image)
            temp += 1
    return gaussian_image



def gaussian_test_nmf(x_hat, y_hat, num_input, image_size, k_features, loss='l2', reg='non', portion_img=1, num_iter=2000):
    """
    :parameter x_hat        input image
    :parameter y_hat        image label
    :parameter num_input    a number of total input images
    :parameter k_features   equals to the number of groups image
    :parameter loss         type of loss function
    :parameter reg          type of regularization
    :parameter portion_img  proportion of training image
    :parameter num_iter     number of iteration
    """

    # create folders / paths
    mkdir('diagram')
    gaussian_image_path = mkdir('Gaussian/gaussian_img/{0}'.format(loss))
    gaussian_recon_path = mkdir('Gaussian/gaussian_recon_img/{0}'.format(loss))
    gaussian_feature_path = mkdir('Gaussian/gaussian_feature/{0}'.format(loss))

    # add Gaussian noise to images
    print("- adding Gaussian noise to images")
    images = add_gaussian_noise(x_hat[:num_input], portion_img=portion_img)

    # save gaussian image to gaussian_img folder
    count = 1

    for j in images:
        gaussian_img = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(gaussian_image_path, count), gaussian_img, cmap='gray')
        count += 1
    print("images with Gaussian noise are saved to {0} folder".format(gaussian_image_path))

    # reshape gaussian image array into a (2016,2414) matrix -> (img.row * img.col, num of img)
    # each col represents an image
    reshape_Gaussian_img = img_array_reshape(images)
    x_hat = img_array_reshape(x_hat)

    # execute NMF algorithm on Gaussian images
    print("- starting NMF algorithm on Gaussian images")
    d, r, rre, error = nmf(x_hat=x_hat[:, :num_input], x=reshape_Gaussian_img, noise_type='gaussian',
                             k_features=k_features, num_iter=num_iter, res=0.0001, loss=loss, reg=reg)

    print("- reconstructing image")
    # reconstruct image
    recon_img = np.dot(d, r)
    count = 1
    for j in recon_img.T:
        pic_1 = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(gaussian_recon_path, count), pic_1, cmap='gray')
        count += 1

    print("- evaluating algorithm accuracy")
    # evaluate the accuracy
    acc, nmi = evaluation(r.T, y_hat[:num_input])

    print("- storing dictionary (features) into gaussian_feature directory")
    # visualize the dictionary (features)
    count = 1
    for j in (d.T):
        pic_1 = np.resize(j, (image_size[1], image_size[0]))
        plt.imsave('{0}/{1}.jpg'.format(gaussian_feature_path, count), pic_1, cmap='gray')
        count += 1

    # save acc,nmi, rre to result.txt
    # [noise type]  [prop]  [loss]  [reg]
    # [loss error]  [rre]   [acc]   [nmi]
    content = "Gaussian - loss type:{0} regularization:{1}\n" \
              "Loss error\t\tRRE\t\t\tACC\t\t\tNMI\n" \
              "{2}\t{3}\t{4}\t{5}\n\n".format(loss, reg, error, rre, acc, nmi)
    write_log(content)



def img_array_reshape(images):
    """
        :parameter image:    image data in [[img1],[img2],[img3]]
        :return: Path
    """
    flatten_images = []
    for i in images:
        i = np.asarray(i).reshape(-1, 1)
        flatten_images.append(i)
    return np.concatenate(flatten_images, axis=1)



def mkdir(path):
    """
    :parameter path:    Path of folders would be created
    :return: Path
    """
    folder = os.path.exists(os.path.join(sys.path[0], path))
    path = os.path.join(sys.path[0], path)
    if not folder:
        os.makedirs(path)

    return path


