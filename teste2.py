#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:16:36 2020

@author: gregorio
"""


from PIL import Image
import cv2 as cv
import argparse
from skimage import exposure
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
pixels=[]

def RGBtoGray(img):

    c= cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    cv.imshow('Gray', c)
    return c
    
    
def Strechting(img):    
    adjusted = cv.equalizeHist(img)
    cv.imshow('Strechting', adjusted)

    
    
def inverted(img):    
    invertida = cv.bitwise_not(img)
    cv.imshow("Inverted",invertida)    
    
    
    
def switch_column(img):
    #baseado no exemplo do William
    rows = img.shape[0]
    columns = img.shape[1]
    aux = []
    
    new_image = np.ndarray((rows,columns), dtype = np.uint8)
    
    for row in range(rows):
        for column in range(columns-1):
            if(column%2) ==0:
                new_image[row][column]= img[row][column+1]
                aux.append(new_image[row][column])
            else:
                new_image[row][column]= img[row][column-1]
                aux.append(new_image[row][column])
            
    cv.imshow("Columns switch", new_image)


    
def switch_row(img):
    #baseado no exemplo do William
    rows = img.shape[0]
    columns = img.shape[1]
    aux = []
    
    new_image = np.ndarray((rows,columns), dtype = np.uint8)
    
    for row in range(rows-1):
        for column in range(columns):
            if(row%2) ==0:
                new_image[row][column]= img[row+1][column]
                aux.append(new_image[row][column])
            else:
                new_image[row][column]= img[row-1][column]
                aux.append(new_image[row][column])
            
    cv.imshow("Rows switch", new_image)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--operation", required=True, help= "Options: 'negative', 'coluns', 'rows', 'strechting' ")
    ap.add_argument("-i", "--path", required=True, help= "Path of the files ")
    args = vars(ap.parse_args())
    

    img = cv.imread(args["path"])
    #img = cv.imread("/home/gregorio/Pictures/nfl.jpg")
        
    gray = RGBtoGray(img)
        
    if(args["operation"] == 'negative'):
        inverted(gray)
    elif(args["operation"] == 'columns'):
        switch_column(gray)
    elif(args["operation"] == 'rows'):
        switch_row(gray)
    elif(args["operation"] == 'strechting'):
        Strechting(gray)
    



    cv.waitKey(0)
    cv.destroyAllWindows()