import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from model_part import Conv,Down,Up
from model import Unet
from cubdl_master.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img

if __name__ == "__main__":
    # From this file, you need to download the PICMUS dataset, then
    # choose the parameters
    phase = 1  # 1 for training sets|2 for testing sets|3 for validation set

    acquisition_type = 2     #-- 1 = simulation || 2 = experiments
    #-- 1 = resolution & distorsion || 2 = contrast & speckle quality 
    #-- 3 = carotid_cross(only experiments)  || 4 = carotid_long(only experiments)
    phantom_type = 4
    data_type = 1              #-- 1 = IQ || 2 = RF
    compound_num = 1
    epoch_num = 0

    if acquisition_type == 1:
        acquisition = "simulation"
    elif acquisition_type == 2:
        acquisition = "experiments"
    else:
        acquisition = "simulation"
    
    if phantom_type == 1:
        phantom = "resolution_distorsion"
    elif phantom_type == 2:
        phantom = "contrast_speckle"
    elif phantom_type == 3:
        phantom = "carotid_cross"
    elif phantom_type == 4:
        phantom = "carotid_long"
    else:
        phantom = "resolution_distorsion"
    
    if data_type == 1:
        data1 = "iq"
    elif data_type == 2:
        data1 = "rf"
    else:
        data1 = "iq"
    
    plane_wave_data = load_datasets(acquisition,phantom,data1)
    angle_num = plane_wave_data.angles.size
    img_dict = {}
    index = 0

    if phase == 1:
        epoch_n = np.arange(60)
        epoch_num = len(epoch_n)
    elif phase == 2:
        epoch_n = np.arange(60,75)
        epoch_num = len(epoch_n)
        epoch_n = np.arange(epoch_num)
    else:
        epoch_n = np.arange(60)


    for i in epoch_n:
        # i1 = i + 60
        das1,iqdata1,xlims,zlims = create_network(plane_wave_data,[i])
        bimg = mk_img(das1,iqdata1)
        if index == 0:
            n,m = bimg.shape
            # img_dict = np.zeros((angle_num+compound_num,n,m))
            img_dict = np.zeros((epoch_num + compound_num, n, m))
            index = 1
        bimg -= np.amin(bimg)
        img_dict[i,:,:] = bimg

    angle_list5 = np.round(np.linspace(0,74,75))

    das5,iqdata5,xlims,zlims = create_network(plane_wave_data,angle_list5)

    bimg5 = mk_img(das5,iqdata5)

    bimg5 -= np.amin(bimg5)
    img_dict[epoch_num,:,:] = bimg5

    # mat_str5 = 'img_data/%d_compound_angles.mat'% len(angle_list5)
    # sio.savemat(mat_str5,{'75_compound_data':bimg5})

    phase_str = ''
    if phase == 1:
        phase_str = '_train'
    elif phase == 2:
        phase_str = '_test'

    mat_str = 'img_data1/'+ acquisition + '_' + phantom + '_' + data1 + phase_str + '.mat'
    name_str = acquisition + '_' + phantom + '_' + data1 + '_data'
    sio.savemat(mat_str,{name_str:img_dict})



    