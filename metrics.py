# File:       metrics.py
# Author:     Tang Jiahua
# Created on: 2021-06-26
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from PixelGrid import make_pixel_grid
import pytorch_ssim
import torch
from sklearn import metrics

# Compute contrast ratio
def contrast(img1, img2):
    return img1.mean() / img2.mean()


# Compute contrast-to-noise ratio
def cnr(img1, img2):
    return np.abs(img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())


# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    a = np.concatenate((img1, img2))
    _, bins = np.histogram(a, bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def MI(img1, img2):
    image1 = np.squeeze(img1)
    image2 = np.squeeze(img2)
    result_NMI = metrics.normalized_mutual_info_score(image1.flatten(), image2.flatten())

    return result_NMI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(np.log2(c_normalized))
    return H

def res_FWHM(img):
    # TODO: Write FWHM code
    raise NotImplementedError


def speckle_res(img):
    # TODO: Write speckle edge-spread function resolution code
    raise NotImplementedError


def snr(img):
    return img.mean() / img.std()


## Compute L1 error
def l1loss(img1, img2):
    return np.abs(img1 - img2).mean()


## Compute L2 error
def l2loss(img1, img2):
    return np.sqrt(((img1 - img2) ** 2).mean())


def psnr(img1, img2):
    dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return 20 * np.log10(dynamic_range / l2loss(img1, img2))


def ncc(img1, img2):
    # return (img1 * img2).sum() / np.sqrt((img1 ** 2).sum() * (img2 ** 2).sum())
    return ((img1-img1.mean()) * (img2-img2.mean())).sum() / np.sqrt(((img1-img1.mean()) ** 2).sum() * ((img2-img2.mean()) ** 2).sum())

def Compute_6dB_Resolution(x_axis, y_signal):
    coeff = 10
    nb_sample = np.size(x_axis)
    nb_interp = nb_sample * coeff
    x_interp = np.linspace(x_axis[0],x_axis[nb_sample-1],nb_interp)
    y_interp = np.interp(x_interp,x_axis,np.squeeze(y_signal))

    ind = np.where(y_interp >= (np.max(y_interp)-6))
    idx1 = np.min(ind)
    idx2 = np.max(ind)
    res = x_interp[idx2] - x_interp[idx1]

    return res

class image_evaluation():
    # image evaluation
    # test_type:  1 simulation point target
    #             2 simulation phantom target
    #             3 experimental point target
    #             4 experiment phantom target
    #             5 in-vivo target
    def __init__(self):
        self.result = {'CR':[], 'CNR':[], 'sSNR':[], 'GCNR':[], 'PSNR':[], 'NCC':[], 'L1loss':[], 'L2loss':[], 'FWHM':[], 'SSIM':[], 'MI':[]}
        self.average_score_FWHM = 0
        self.average_score_GCNR = 0
        self.average_score_sSNR = 0
        self.average_score_CNR = 0
        self.average_score_CR = 0

    def evaluate(self,img1,img2,opt,plane_wave_data, test_type, i):
        img1 -= np.max(img1)
        img2 -= np.max(img2)

        if test_type == 2:
            self.occlusionDiameter = np.array([0.008, 0.008,0.008,0.008,0.008,0.008,0.008,0.008,0.008])
            self.r = 0.004
            self.rin = self.r - 6.2407e-4
            self.rout1 = self.r + 6.2407e-4
            self.rout2 = 1.2 * np.sqrt(self.rin*self.rin + self.rout1*self.rout1)
            self.xcenter = [0,0,0,-0.012,-0.012,-0.012,0.012,0.012,0.012]
            self.zcenter = [0.018, 0.03, 0.042, 0.018, 0.03, 0.042, 0.018, 0.03, 0.042]

        if test_type == 4:
            self.occlusionDiameter = np.array([0.0045, 0.0045])
            self.r = 0.0022
            self.rin = self.r - 6.2407e-4
            self.rout1 = self.r + 6.2407e-4
            self.rout2 = 1.2 * np.sqrt(self.rin * self.rin + self.rout1 * self.rout1)
            if i <=30:
                self.xcenter = [-1.0e-04, -1.0e-04]
                self.zcenter = [0.0149, 0.0428]
            if i>30 and i <=45:
                self.xcenter = [1.0e-04, 1.0e-04]
                self.zcenter = [0.0149, 0.0428]
            if i >45:
                self.xcenter = [-1.0e-04, -1.0e-04]
                self.zcenter = [0.0172, 0.0451]

        xlims = [plane_wave_data.ele_pos[0, 0], plane_wave_data.ele_pos[-1, 0]]
        zlims = [5e-3, 55e-3]
        wvln = plane_wave_data.c / plane_wave_data.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        grid = make_pixel_grid(xlims, zlims, dx, dz)
        self.x_matrix = grid[:,:,0]
        self.z_matrix = grid[:,:,2]
        value = 0
        image = img1

        if test_type == 1 or test_type == 3:
            # FWHM
            # image = img1
            maskROI = np.zeros((508,387))
            if test_type == 1 and i <= 45:
                sca = np.array([[0,0,0.01],[0,0,0.015],[0,0,0.02],[0,0,0.025],[0,0,0.03],[0,0,0.035],[0,0,0.04],[0,0,0.045],
                        [-0.015,0,0.02],[-0.01,0,0.02],[-0.005,0,0.02],[0.005,0,0.02],[0.01,0,0.02],[0.015,0,0.02],[-0.015,0,0.04],
                        [-0.01,0,0.04],[-0.005,0,0.04],[0.005,0,0.04],[0.01,0,0.04],[0.015,0,0.04]])
            if test_type == 1 and i > 45:
                sca = np.array([[0, 0, 0.015], [0, 0, 0.02], [0, 0, 0.025], [0, 0, 0.03], [0, 0, 0.035],
                                [0, 0, 0.04], [0, 0, 0.045], [0, 0, 0.05],
                                [-0.015, 0, 0.02], [-0.01, 0, 0.02], [-0.005, 0, 0.02], [0.005, 0, 0.02],
                                [0.01, 0, 0.02], [0.015, 0, 0.02], [-0.015, 0, 0.04],
                                [-0.01, 0, 0.04], [-0.005, 0, 0.04], [0.005, 0, 0.04], [0.01, 0, 0.04],
                                [0.015, 0, 0.04]])
            if test_type == 3 :
                if i <= 30:
                    sca = np.array([[-0.0005,0,0.0096],[-0.0004,0,0.0187],[-0.0004,0,0.028],[-0.0002,0,0.0376],[-0.0001,0,0.047],
                                [-0.0105,0,0.0375],[0.0098,0,0.0376]])
                if i > 30 and i <= 45:
                    sca = np.array(
                        [[0.0005, 0, 0.0096], [0.0004, 0, 0.0187], [0.0004, 0, 0.028], [0.0002, 0, 0.0376],
                         [0.0001, 0, 0.047],
                         [0.0105, 0, 0.0375], [-0.0098, 0, 0.0376]])
                if i > 45:
                    sca = np.array(
                        [[-0.0005, 0, 0.0504], [-0.0004, 0, 0.0413], [-0.0004, 0, 0.032], [-0.0002, 0, 0.0224],
                         [-0.0001, 0, 0.013],
                         [-0.0105, 0, 0.0225], [0.0098, 0, 0.0224]])
            for k in range(sca.shape[0]):
                x = sca[k][0]
                z = sca[k][2]
                mask = (k+1) * ((self.x_matrix > (x-0.0018)) & (self.x_matrix < (x+0.0018)) & (self.z_matrix > (z-0.0018))& (self.z_matrix<(z+0.0018)))
                maskROI = maskROI + mask
            patchImg1 = np.zeros((508,387))

            patchImg1[0:508,0:384] = image[0][0:508,:]
            patchImg1[:,384:387] = patchImg1[:,381:384]


            score1 = np.zeros((sca.shape[0],2))

            for k in range(sca.shape[0]):
                patchMask = np.copy(maskROI)
                patchImg = np.copy(patchImg1)
                patchImg[maskROI != (k+1)] = np.min(np.min(min(image)))
                patchMask[maskROI != (k+1)] = 0
                [idzz, idxx] = np.where(patchMask == (k+1))

                x_lim_patch = np.array([plane_wave_data.x_axis[np.min(idxx)],plane_wave_data.x_axis[np.max(idxx)]]) *1e3
                z_lim_patch = np.array([plane_wave_data.z_axis[np.min(idzz)], plane_wave_data.z_axis[np.max(idzz)]]) * 1e3
                a = np.arange(np.min(idxx),np.max(idxx)+1)
                x_patch = plane_wave_data.x_axis[a] *1e3
                b = np.arange(np.min(idzz),np.max(idzz)+1)
                z_patch = plane_wave_data.z_axis[b] *1e3

                [idz,idx] = np.where(patchImg == np.max(np.max(patchImg)))
                signalLateral = patchImg[idz,np.min(idxx):(np.max(idxx)+1)]
                signalAxial = patchImg[np.min(idzz):(np.max(idzz)+1),idx]
                # a = signalLateral.reshape(-1,1)
                # plt.plot(a)

                res_axial = Compute_6dB_Resolution(z_patch,signalAxial)
                res_lateral = Compute_6dB_Resolution(x_patch,signalLateral)
                score1[k][0] = res_axial
                score1[k][1] = res_lateral
            self.average_score_FWHM = np.mean(score1)

        if test_type == 2 or test_type == 4:
            # contrast
            score2 = np.zeros(self.occlusionDiameter.shape[0])
            score3 = np.zeros(self.occlusionDiameter.shape[0])
            score4 = np.zeros(self.occlusionDiameter.shape[0])
            score5 = np.zeros(self.occlusionDiameter.shape[0])
            for k in range(len(self.occlusionDiameter)):
                xc = self.xcenter[k]
                zc = self.zcenter[k]
                maskOcclusion = (np.power(self.x_matrix-xc,2) + np.power(self.z_matrix-zc,2)) <= (self.r * self.r)
                maskInside =  (np.power(self.x_matrix-xc,2) + np.power(self.z_matrix-zc,2)) <= (self.rin * self.rin)
                a =  (np.power(self.x_matrix-xc,2) + np.power(self.z_matrix-zc,2)) >= (self.rout1 * self.rout1)
                b = (np.power(self.x_matrix-xc,2) + np.power(self.z_matrix-zc,2)) <= (self.rout2 * self.rout2)
                maskOutside = a&b

                inside = []
                outside = []
                num1 = 0
                num2 = 0
                for i in range(508):
                    for j in range(384):
                        if maskInside[i][j] == True:
                            inside.append(image[0][i][j])
                            num1 += 1
                        if maskOutside[i][j] == True:
                            outside.append(image[0][i][j])
                            num2 += 1
                outside = np.array(outside)
                inside = np.array(inside)
                ll1 = np.mean(inside)
                ll2 = np.mean(outside)
                l1 = np.abs(np.mean(inside)-np.mean(outside))
                l2 = np.sqrt((np.var(inside)+np.var(outside))/2)

                CR1 = np.abs(np.mean(inside)-np.mean(outside))
                CNR = np.abs(np.mean(inside) - np.mean(outside)) / np.sqrt((inside.var() + outside.var()))
                CR2 = 20*np.log10(np.abs(np.mean(inside) - np.mean(outside)) / np.sqrt((np.var(inside) + np.var(outside))/2))
                CR3 = 20*np.log10(np.mean(outside)/np.mean(inside))
                sSNR = np.abs(np.mean(outside)) / np.std(outside)
                GCNR = gcnr(inside, outside)

                score2[k] = CR1
                score3[k] = CNR
                score4[k] = sSNR
                score5[k] = GCNR
                self.average_score_CR = np.mean(score2)
                self.average_score_CNR = np.mean(score3)
                self.average_score_sSNR = np.mean(score4)
                self.average_score_GCNR = np.mean(score5)

                # print(CNR)
        self.PSNR = psnr(img1, img2)
        self.MI = MI(img1, img2)

        ima1 = torch.from_numpy(img1)
        ima2 = torch.from_numpy(img2)
        ima1 = torch.unsqueeze(ima1, 1)
        ima2 = torch.unsqueeze(ima2, 1)
        self.SSIM = pytorch_ssim.ssim(ima1, ima2)
        #print(CR1)
        # print("average:" + str(value/9))

        # self.CR = np.abs(20 * np.log10(contrast(img1, img2)))
        # self.CNR = cnr(img1, img2)
        # self.SNR = snr(img1)
        # self.GCNR = gcnr(img1, img2)
        self.L1Loss = l1loss(img1, img2)
        self.L2Loss = l2loss(img1, img2)
        # self.PSNR = psnr(img1, img2)
        self.NCC = ncc(img1, img2)

        # Restore the results
        self.result['FWHM'].append(self.average_score_FWHM)
        self.result['CR'].append(self.average_score_CR)
        self.result['CNR'].append(self.average_score_CNR)
        self.result['sSNR'].append(self.average_score_sSNR)
        self.result['GCNR'].append(self.average_score_GCNR)
        self.result['L1loss'].append(self.L1Loss)
        self.result['L2loss'].append(self.L2Loss)
        self.result['PSNR'].append(self.PSNR)
        self.result['NCC'].append(self.NCC)
        self.result['SSIM'].append(self.SSIM)
        self.result['MI'].append(self.MI)

        # self.print_results(opt)


    def print_results(self,opt):
        message = ''
        message += '----------------- Evaluations ---------------\n'
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format('FWHM:', str(np.mean(self.result['FWHM'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('Contrast [db]:', str(np.mean(self.result['CR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('CNR:', str(np.mean(self.result['CNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('sSNR:', str(np.mean(self.result['sSNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('GCNR:', str(np.mean(self.result['GCNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('L1 loss:', str(np.mean(self.result['L1loss'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('L2 loss:', str(np.mean(self.result['L2loss'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('PSNR:', str(np.mean(self.result['PSNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('SSIM:', str(np.mean(self.result['SSIM'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('MI:', str(np.mean(self.result['MI'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('NCC:', str(np.mean(self.result['NCC'])), comment)
        message += '----------------- End -------------------'
        # self.save_result(opt,message)
        print(message)


    def save_result(self,opt,message):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        file_name = os.path.join(expr_dir, 'result.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

# if __name__ == "__main__":
#     img1 = np.random.rayleigh(2, (80, 50))
#     img2 = np.random.rayleigh(1, (80, 50))
#     a = image_evaluation()
#     a.evaluate(img1,img2)
#     a.print_results()
    # print("Contrast [dB]:  %f" % (20 * np.log10(contrast(img1, img2))))
    # print("CNR:            %f" % cnr(img1, img2))
    # print("SNR:            %f" % snr(img1))
    # print("GCNR:           %f" % gcnr(img1, img2))
    # print("L1 Loss:        %f" % l1loss(img1, img2))
    # print("L2 Loss:        %f" % l2loss(img1, img2))
    # print("PSNR [dB]:      %f" % psnr(img1, img2))
    # print("NCC:            %f" % ncc(img1, img2))
