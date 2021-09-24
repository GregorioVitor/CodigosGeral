#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:18:26 2020

@author: gregorio
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import pi, exp, sqrt

def rgb_to_gray(image, form=None):

    rows = image.shape[0]
    columns = image.shape[1]

    new_image = np.ndarray((rows, columns), dtype=np.uint8)

    for row in range(rows):

        for column in range(columns):

            b, g, r = image[row][column]
            if form == 'div':
                gray_pixel = int((b + g + r) / 3)
            if form == 'mux':
                gray_pixel = int(0.2989 * r) + int(0.5870 * g) + int(0.1140 * b)

            new_image[row][column] = gray_pixel

    return new_image



def get_gaussian():
    s, k = 1, 2  # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)]
    kernel = np.outer(probs, probs)
    return kernel

def filtering(image, method):

    '''
    this method gets the image and apply the median filter for noise resolution
    kernel size = [3,3]
    border values ignored for the example
    :param image:
    :return image filtered:
    '''
    rows = image.shape[0]
    columns = image.shape[1]

    new_image = np.ndarray((rows-1, columns-1), dtype=np.uint8)

    for row in range(1, rows - 1):

        for column in range(1, columns - 1):

            kernel = np.ones((3, 3))

            if method == 'median':

                kernel[0][0] = image[row - 1][column - 1] * kernel[0][0]
                kernel[0][1] = image[row - 1][column] * kernel[0][1]
                kernel[0][2] = image[row - 1][column + 1] * kernel[0][2]
                kernel[1][0] = image[row][column - 1] * kernel[1][0]
                kernel[1][1] = image[row][column] * kernel[1][1]
                kernel[1][2] = image[row][column + 1] * kernel[1][2]
                kernel[2][0] = image[row + 1][column - 1] * kernel[2][0]
                kernel[2][1] = image[row + 1][column] * kernel[2][1]
                kernel[2][2] = image[row + 1][column + 1] * kernel[2][2]

                median = np.median(kernel.tolist())
                new_image[row][column] = median

            if method == 'mean':

                kernel[0][0] = image[row - 1][column - 1] * kernel[0][0]
                kernel[0][1] = image[row - 1][column] * kernel[0][1]
                kernel[0][2] = image[row - 1][column + 1] * kernel[0][2]
                kernel[1][0] = image[row][column - 1] * kernel[1][0]
                kernel[1][1] = image[row][column] * kernel[1][1]
                kernel[1][2] = image[row][column + 1] * kernel[1][2]
                kernel[2][0] = image[row + 1][column - 1] * kernel[2][0]
                kernel[2][1] = image[row + 1][column] * kernel[2][1]
                kernel[2][2] = image[row + 1][column + 1] * kernel[2][2]

                mean = np.mean(kernel.tolist())
                new_image[row][column] = mean

            if method == 'gaussian':

                kernel = np.ndarray((3, 3))

                kernel[0][0] = int(round(image[row - 1][column - 1] * (1/16)))
                kernel[0][1] = int(round(image[row - 1][column] * (1/8)))
                kernel[0][2] = int(round(image[row - 1][column - 1] * (1/16)))
                kernel[1][0] = int(round(image[row - 1][column] * (1/8)))
                kernel[1][1] = int(round(image[row][column] * (1/4)))
                kernel[1][2] = int(round(image[row - 1][column] * (1/8)))
                kernel[2][0] = int(round(image[row - 1][column - 1] * (1/16)))
                kernel[2][1] = int(round(image[row - 1][column] * (1/8)))
                kernel[2][2] = int(round(image[row - 1][column - 1] * (1/16)))

                aux = []

                for lst in kernel:

                    for value in lst:
                        aux.append(value)

                new_pixel = sum(aux)
                new_image[row][column] = new_pixel



    cv.imshow('Image with {} method'.format(method), new_image)
    cv.waitKey(2000)  # wait 2 seconds showing the image
    #cv.imwrite('results/image_with_{}_method.png'.format(method), new_image)
    return new_image


def main():
    image = cv.imread('/home/gregorio/Desktop/imagens/BoxPlotRatio.png')
    #image = cv.resize(image, (255, 255))
    gray_image = rgb_to_gray(image, 'mux')
    print('Gray Image created')
    print('Creating median Filter Image')
    median_image = filtering(gray_image, 'median')
    print('Median filter Image created')
    print('Creating mean Filter Image')
    

    


if __name__ == '__main__':
    main()