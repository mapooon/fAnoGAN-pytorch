"""Main"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.ion()
import os

from dataset.dataset_class import OneClassMnistDataset
# from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
# from loss.loss_generator import *
# from network.blocks import *
from network.model import *
from tqdm import tqdm

import sys
import torchvision

from params.test_params import path_to_pt,batch_size, z_size, img_size

"""Create dataset and net"""
display_training = False
device = torch.device("cuda:0")
cpu = torch.device("cpu")



netG = nn.DataParallel(Generator(z_size = z_size).to(device))
# netC = nn.DataParallel((z_size=z_size).to(device))
netD = nn.DataParallel(Discriminator().to(device))

netE = nn.DataParallel(Encoder(z_size = z_size, dim = 16, n_channels = 1).to(device))




"""Criterion"""
# criterionG = LossG()
# criterionDreal = LossReal()
# criterionDfake = LossFake()
# criterionC = LossC()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
lossesE = []
i_batch_current = 0

num_epochs = 30


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_pt, map_location=cpu)
netG.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
netD.module.load_state_dict(checkpoint['D_state_dict'])
netE.module.load_state_dict(checkpoint['E_state_dict'])
# i_batch_current = checkpoint['i_batch'] +1
# optimizerE.load_state_dict(checkpoint['optimizerE'])

netG.eval()
netD.eval()
netE.eval()

"""Training"""
batch_start = datetime.now()
if not display_training:
    matplotlib.use('agg')

# one = torch.FloatTensor([1]).to(device)
# mone = one * -1
for i in range(10):
    dataset = OneClassMnistDataset(number=i,img_size=img_size,phase='test')
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=16,
                            pin_memory=True,
                            drop_last = True)


    pbar = tqdm(dataLoader, leave=True, initial=0)
    # pbar.set_postfix(epoch=epoch)
    A_R_total = 0.0
    A_D_total = 0.0
    A_x_total = 0.0
    for i_batch, data in enumerate(pbar):
        
        x_real = data.to(device).float()
        
        if i_batch % 1 == 0:

            #train D
            with torch.no_grad():

                z = netE(x_real)
                
                x_fake = netG(z)

                feat_fake = netD(x_fake,return_feat=True)

                feat_real = netD(x_real,return_feat=True)

                if i_batch == 0:
                    A_R = nn.MSELoss(reduction='none')(feat_fake,feat_real)
                    A_D = nn.MSELoss(reduction='none')(x_fake,x_real)
                    A_R_save = A_R.view(len(A_R),-1).mean(1)
                    # A_R_save = A_R_save[:2].to('cpu').data.numpy().tolist()
                    A_D_save = A_D.view(len(A_D),-1).mean(1)
                    # A_D_save = A_D_save[:2].to('cpu').data.numpy().tolist()
                    A_x_save = A_R_save + A_D_save
                    A_x_save = A_x_save.to('cpu').data.numpy().tolist()
                    A_R = nn.MSELoss()(feat_fake,feat_real)
                    A_D = nn.MSELoss()(x_fake,x_real)
                else:
                    A_R = nn.MSELoss()(feat_fake,feat_real)
                    A_D = nn.MSELoss()(x_fake,x_real)

                A_x = A_R + A_D
                A_x_total += A_x.item()
                A_D_total += A_D.item()
                A_R_total += A_R.item()
                # gradient_penalty = calc_gradient_penalty(netD,x_real,x_fake)
                # gradient_penalty.backward()

                
                #for p in D.module.parameters():
                #   p.data.clamp_(-1.0, 1.0)
                if i_batch == 0:
                    # torchvision.utils.save_image(torch.cat([x_real[:2],x_fake[:2],(x_real[:2]-x_fake[:2]).abs()],dim=0), f'{i}_{A_R_save[0]:.4f}+{A_D_save[0]:.4f}_{A_R_save[1]:.4f}+{A_D_save[1]:.4f}.png', nrow=2, normalize=True, range=(0, 1))
                    torchvision.utils.save_image(torch.cat([x_real[:2],x_fake[:2],(x_real[:2]-x_fake[:2]).abs()],dim=0), f'{i}_{A_x_save[0]:.4f}_{A_x_save[1]:.4f}.png', nrow=2, normalize=True, range=(0, 1))
    A_x_result = A_x_total/len(dataLoader)
    A_R_result = A_R_total/len(dataLoader)
    A_D_result = A_D_total/len(dataLoader)
    print(f'{i}: Mean A_x = {A_x_result:4f}, Mean A_R = {A_R_result:4f}, Mean A_D = {A_D_result:4f}')
                    

       