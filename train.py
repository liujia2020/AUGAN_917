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
from thop import profile
# from model_part import Conv,Down,Up
# from model import Unet
from cubdl_master.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img,PlaneWaveData
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
    plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
    das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])

    opt = TrainOptions().parse() # get training options

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
    # lossG = np.zeros(opt.n_epochs+opt.niter_decay)
    # lossD = np.zeros(opt.n_epochs + opt.niter_decay)
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
        for data, target1, target2, target3, target4, target5 in train_bar:
            iter_start_time = time.time()  # timer for computation per iteration
            # if total_iters % opt.print_freq == 0:
            # t_data = iter_start_time - iter_data_time
            n1,n2,n3 = data.shape

            # test_image(target5[0],xlims,zlims,1)
            # test_image(data[0],xlims,zlims,2)

            data = torch.reshape(data,(1,1,n2,n3))
            target5 = torch.reshape(target5,(1,1,n2,n3))#
            net = model.netG.to(model.device)
            data = data.to(model.device)

            # total_iters = total_iters + opt.batch_size
            epoch_iter = epoch_iter + opt.batch_size
            model.set_input(data,target5)
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            # if epoch % 100 == 0:
            #     test_image(model.real_A[0].cpu(),model.fake_B[0].cpu(),model.real_B[0].cpu(),xlims,zlims,epoch)

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

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

        # img_eva.evaluate(model.fake_B[0].cpu().detach().numpy(), model.real_B[0].cpu().detach().numpy(), opt)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        #if epoch % 1 == 0:
            #test_image(model.real_A[0].cpu(), model.fake_B[0].cpu(), model.real_B[0].cpu(), xlims, zlims, epoch, opt.phase, opt.name)
            #print(loss_beta)

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
    
