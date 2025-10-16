import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import time
try:
    from thop import profile
    
except Exception:
    profile = None  # optional; only used if FLOPs profiling is needed
# from model_part import Conv,Down,Up
# from model import Unet
from cubdl_master.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img
from options.train_options import TrainOptions
from models import create_model
from data_process import load_dataset, test_image
from util.util import diagnose_network
from models.network import UnetGenerator
#import PlaneWaveData
import math
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage,CenterCrop,Resize,RandomRotation,RandomVerticalFlip,RandomHorizontalFlip,Resize,ColorJitter
from metrics import image_evaluation
# import thop

def get_args(): # initial options for the training
    parser = argparse.ArgumentParser(description="my code:")
    parser.add_argument('-b',"--batch_size",type=int,default=10, help="batch size each train epoch")
    parser.add_argument('-n',"--num_epoch",type=int,default=100,help="training epoch numbers")
    parser.add_argument('-l',"--learning_rate",type=float,default=0.0001,help="learing rate")
    parser.add_argument('-f', '--load', type=str, default='./img_data',
                        help='Load model from a file')
    parser.add_argument('-s','--scale',type=float,default=0.5,help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation',type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()

def makedir(): # Making the directory for saving pictures of loss change.
    train_base = './images/' + opt.name + '/train'
    test_base = './images/' + opt.name + '/test'
    if not os.path.exists(train_base):
        os.makedirs(train_base)
    if not os.path.exists(test_base):
        os.makedirs(test_base)
    loss_path = './images/' + opt.name + '/train/loss.png'
    return loss_path

if __name__ == '__main__':
    # Initial setting and correponding parameters
    # PICMUS dataset is only used to get x/z limits for visualization; fall back if missing
    try:
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
    except Exception as e:
        logging.warning("PICMUS dataset not available, skip visualization setup: %s", e)
        plane_wave_data = None
        xlims, zlims = [0, 1], [0, 1]

    opt = TrainOptions().parse() # 解析命令行参数

    # Load the model
    # opt.continue_train = True # whether continue to training
    model = create_model(opt)
    model.setup(opt)
    total_iters = 0    # the total number of training iterations
    loss_G = 0
    loss_G1 = 0
    loss_D = 0

    # Displaying the options.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:  %(message)s')
    # opt = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device  {device}')
    loss_path = makedir()

    # Loading the dataset
    img_dataset = load_dataset(opt, opt.phase, 0)
    dataset_len = img_dataset.len
    train_loader = DataLoader(dataset=img_dataset, num_workers=0, batch_size=1, shuffle=True)
    lossG = np.zeros(opt.n_epochs+opt.niter_decay)
    lossD = np.zeros(opt.n_epochs+opt.niter_decay)

    # Image evaluation
    img_eva = image_evaluation()
    loss_beta = np.zeros((100))
    index = 0

    # Training process
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.niter_decay +1):
        epoch_start_time = time.time() # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        train_bar = tqdm(train_loader)

        for input_image, target_image in train_bar:
            iter_start_time = time.time()  # timer for computation per iteration

            n1,n2,n3 = input_image.shape
            input_image = torch.reshape(input_image,(1,1,n2,n3)) # batch_size, channel, height, width
            target_image = torch.reshape(target_image,(1,1,n2,n3))  # reshape target to NCHW
            # 将数据送入模型，执行优化
            net = model.netG.to(model.device)
            input_image = input_image.to(model.device)

            # total_iters = total_iters + opt.batch_size
            epoch_iter = epoch_iter + opt.batch_size
            
            model.set_input(input_image, target_image)
            model.optimize_parameters()  

            if total_iters % opt.print_freq == 0:
                train_bar.set_description(
                    desc='[converting LR images to SR images] lossG: %.4f , lossD: %.4f' % (
                        model.loss_G, model.loss_D))

            loss_G += model.loss_G
            loss_D += model.loss_D
            loss_G1 += model.loss_G
            total_iters = total_iters + 1

            if total_iters % 50 == 0:
                # model.save_networks(epoch)
                if index != 20:
                    loss_beta[index] = loss_G1 / 50
                    index = index + 1
                    loss_G1 = 0
                    #print(loss_beta)
                elif index == 20:
                    sio.savemat('beta_data.mat',{'data':loss_beta})
                    aaaa = 1

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


        lossG[epoch-1] = loss_G / dataset_len
        lossD[epoch-1] = loss_D / dataset_len
        loss_G = 0
        loss_D = 0
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        print(lossG)

    plt.subplot(121)
    plt.plot(lossG)
    plt.title("lossG figure" , fontsize=10)
    plt.subplot(122)
    plt.plot(lossD)
    plt.title("lossD figure", fontsize=10)
    plt.savefig(loss_path)
    plt.show()