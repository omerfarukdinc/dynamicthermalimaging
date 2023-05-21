# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:51:26 2023

@author: OmerFarukDinc
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
from pathlib import Path
import pandas as pd
from scipy.signal import chirp, find_peaks, peak_widths
x_1pixel=0.188301 #Pixel Size in real world
y_1pixel=0.184881
def fwhm(y):
    peaks,_=find_peaks(y)
    result=peak_widths(y,peaks,rel_height=0.5)
    return result,peaks
fwhm_liste_tumor=list()
fwhm_liste_saglıklı=list()
def find_max_pixel(array):
    w,h=array.shape
    pixel_list=list()
    coor=list()
    max_array=np.max(array)
    for i in range(w):
        for j in range(h):
            pixel_array=array[i,j]
            if pixel_array==max_array:
                pixel_list.append(pixel_array)
                coor.append([i,j])
    
    return coor,pixel_list
def fwhm_veriler(path):
    path_given=Path(path)
    given_liste=list(path_given.glob("*"))
    Path_Series = pd.Series(given_liste,dtype='str',name="PATH")
    fwhm_liste_given_x=list()
    fwhm_liste_given_y=list()
    fwhm_liste_xy=list()
    fwhm_liste_yx=list()
    peaks_liste_xy=list()
    peaks_liste_yx=list()

    peaks_given_list_x=list()
    peaks_given_list_y=list()
    for i in range(len(given_liste)):
        
        tumor_fare=cv2.imread(str(given_liste[i]),0)
        tumor_fare=tumor_fare[190:250,348-30:348+35]#[171:232,247:315]200:250,348-30:348+25 #Region of interest of thermal image
        
        koordinat_hot,value_hot=find_max_pixel(tumor_fare)
        x_1=koordinat_hot[0][0]
        y_1=koordinat_hot[0][1]
        # x_1=127
        # y_1=157
        diag=np.diagonal(tumor_fare[x_1-18:x_1+18,y_1-18:y_1+18])
        diag_flip=np.diagonal(np.fliplr(tumor_fare[x_1-18:x_1+18,y_1-18:y_1+18]))
        tumor_fwhm_cprz_xy,peaks_cprz_xy=fwhm(diag)#soldan sağa eksen\
        tumor_fwhm_cprz_yx,peaks_cprz_xy=fwhm(diag_flip)#/

        tumor_fwhm,peaks_given=fwhm(tumor_fare[x_1-18:x_1+18,y_1])
        tumor_fwhm_y,peaks_given_y=fwhm(tumor_fare[x_1-18:x_1+18,y_1])
        tumor_fwhm_x,peaks_given_x=fwhm(tumor_fare[x_1,y_1-18:y_1+18])
        peaks_given_list_y.append(np.max(tumor_fare[x_1-18:x_1+18,y_1]))
        peaks_liste_xy.append(np.max(diag))
        peaks_liste_yx.append(np.max(diag_flip))

        peaks_given_list_x.append(np.max(tumor_fare[x_1,y_1-18:y_1+18]))
        
        max_fwhm_xy=list()
        for q in range(len(tumor_fwhm_cprz_xy[2])):
            fwhm_given=abs(tumor_fwhm_cprz_xy[2][q]-tumor_fwhm_cprz_xy[3][q])
            max_fwhm_xy.append(fwhm_given)
        max_fwhm_yx=list()

        for j in range(len(tumor_fwhm_cprz_yx[2])):
            fwhm_given=abs(tumor_fwhm_cprz_yx[2][j]-tumor_fwhm_cprz_yx[3][j])
            max_fwhm_yx.append(fwhm_given)
        fwhm_liste_xy.append(np.max(max_fwhm_xy))
        fwhm_liste_yx.append(np.max(max_fwhm_yx))
        
        
        max_fwhm_x=list()
        for q in range(len(tumor_fwhm_x[2])):
            fwhm_given=abs(tumor_fwhm_x[2][q]-tumor_fwhm_x[3][q])
            max_fwhm_x.append(fwhm_given)
        max_fwhm_y=list()

        for j in range(len(tumor_fwhm_y[2])):
            fwhm_given=abs(tumor_fwhm_y[2][j]-tumor_fwhm_y[3][j])
            max_fwhm_y.append(fwhm_given)
        fwhm_liste_given_x.append(np.max(max_fwhm_x))
        fwhm_liste_given_y.append(np.max(max_fwhm_y))

    given_peaks_x=pd.Series(peaks_given_list_x,dtype="int",name="Peaks_x")
    given_peaks_y=pd.Series(peaks_given_list_y,dtype="int",name="Peaks_y")
    
    given_Series_y = pd.Series(fwhm_liste_given_y,dtype='float',name="FWHM_y")
    given_Series_x = pd.Series(fwhm_liste_given_x,dtype='float',name="FWHM_x")
    
    given_peaks_xy=pd.Series(peaks_liste_xy,dtype="int",name="Peaks_xy")
    given_peaks_yx=pd.Series(peaks_liste_yx,dtype="int",name="Peaks_yx")
    
    given_Series_yx = pd.Series(fwhm_liste_yx,dtype='float',name="FWHM_yx")
    given_Series_xy = pd.Series(fwhm_liste_xy,dtype='float',name="FWHM_xy")
    given_Data = pd.concat([given_Series_x,given_Series_y,Path_Series,given_peaks_y,given_peaks_x,given_peaks_xy,given_peaks_yx,given_Series_yx,given_Series_xy],axis=1)
    return given_Data 
