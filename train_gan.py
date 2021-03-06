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

from params.train_gan_params import path_to_chkpt, path_to_backup, batch_size, z_size, img_size

"""Create dataset and net"""
display_training = False
device = torch.device("cuda:0")
cpu = torch.device("cpu")
dataset = OneClassMnistDataset(number=0,img_size=img_size)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=16,
                        pin_memory=True,
                        drop_last = True)

netG = nn.DataParallel(Generator(z_size=z_size).to(device))
# netC = nn.DataParallel((z_size=z_size).to(device))
netD = nn.DataParallel(Discriminator().to(device))

netG.train()

netD.train()


optimizerG = optim.Adam(params = netG.parameters(),
                        lr=1e-4,
                        betas=(0.0,0.9))
optimizerD = optim.Adam(params = netD.parameters(),
                        lr=1e-4,
                        betas=(0.0,0.9))

"""Criterion"""
# criterionG = LossG()
# criterionDreal = LossReal()
# criterionDfake = LossFake()
# criterionC = LossC()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 100000

#initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    netG.apply(init_weights)
    netD.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'G_state_dict': netG.module.state_dict(),
            'D_state_dict': netD.module.state_dict(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
netG.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
netD.module.load_state_dict(checkpoint['D_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
i_batch_current = checkpoint['i_batch'] +1
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])

netG.train()
netD.train()

"""Training"""
batch_start = datetime.now()
pbar = tqdm(dataLoader, leave=True, initial=0)
if not display_training:
    matplotlib.use('agg')

one = torch.FloatTensor([1]).to(device)
mone = one * -1

for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)
    pbar.set_postfix(epoch=epoch)
    for i_batch, data in enumerate(pbar, start=0):
        
        x_real = data.to(device).float()
        
        if i_batch % 1 == 0:

            #train D
            with torch.autograd.enable_grad():
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                netG.zero_grad()
                netD.zero_grad()

                z = torch.randn((len(x_real),z_size)).to(device)
                
                with torch.no_grad():
                    x_fake = netG(z)

                r_fake = netD(x_fake)
                lossDfake = nn.MSELoss()(r_fake,torch.ones_like(r_fake))#criterionDfake(r_fake)

                r = netD(x_real)
                lossDreal = nn.MSELoss()(r,-torch.ones_like(r))#criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward()

                # gradient_penalty = calc_gradient_penalty(netD,x_real,x_fake)
                # gradient_penalty.backward()

                optimizerD.step()
                #for p in D.module.parameters():
                #   p.data.clamp_(-1.0, 1.0)
                
    # train G
    with torch.autograd.enable_grad():
        #zero the parameter gradients
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        netG.zero_grad()
        netD.zero_grad()

        z = torch.randn((len(x_real),z_size)).to(device)

        x_fake = netG(z)

        r_fake = netD(x_fake)

        lossG = nn.MSELoss()(r_fake,-torch.ones_like(r_fake))#-r_fake.mean()#criterionG(r_fake)
        
        lossG.backward(retain_graph=False)
        optimizerG.step()
        #optimizerD.step()
        
            
                    

        # Output training stats
        if i_batch % 1 == 0 and i_batch > 0:
            #batch_end = datetime.now()
            #avg_time = (batch_end - batch_start) / 100
            # print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
            
            #batch_start = datetime.now()
            
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
            #       % (epoch, num_epochs, i_batch, len(dataLoader),
            #          lossD.item(), lossG.item(), r.mean(), r_hat.mean()))
            pbar.set_postfix(epoch=epoch, r=r.mean().item(), rhat=r_fake.mean().item(), lossG=lossG.item())

            if display_training:
                plt.figure(figsize=(10,10))
                plt.clf()
                out = (y_hat[0]*255).transpose(0,2)
                for img_no in range(1,y_hat.shape[0]//16):
                    out = torch.cat((out, (y_hat[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = out

                plt.clf()
                out = (x[0]*255).transpose(0,2)
                for img_no in range(1,x.shape[0]//16):
                    out = torch.cat((out, (x[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = np.concatenate((fig, out), 0)

                plt.clf()
                out = (l_y[0]*255).transpose(0,2)
                for img_no in range(1,l_y.shape[0]//16):
                    out = torch.cat((out, (l_y[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                
                fig = np.concatenate((fig, out), 0)
                plt.imshow(fig)
                plt.xticks([])
                plt.yticks([])
                plt.draw()
                plt.pause(0.001)
            
            

        if i_batch % 1000 == 999:
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

            if display_training:
                plt.clf()
                plt.plot(lossesG) #blue
                plt.plot(lossesD) #orange
                plt.plot(lossesC) #green
                plt.show()

            print('Saving latest...')
            torch.save({
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'G_state_dict': netG.module.state_dict(),
                    'D_state_dict': netD.module.state_dict(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, path_to_chkpt)
            # torchvision.utils.save_image(x, 'recent_x.png', nrow=len(x), normalize=True, range=(0, 1))
            # torchvision.utils.save_image(y, 'recent_y.png', nrow=len(y), normalize=True, range=(0, 1))
            # torchvision.utils.save_image(l_y.unsqueeze(1).repeat(1,3,1,1), 'recent_l_y.png', nrow=len(l_y), normalize=True, range=(0, 1))
            torchvision.utils.save_image(x_fake[:8], 'recent_train_gan.png', nrow=8, normalize=True, range=(0, 1))
            print('...Done saving latest')
            
    if epoch%1 == 0:
        print('Saving latest...')
        torch.save({
                'epoch': epoch+1,
                'lossesG': lossesG,
                'lossesD': lossesD,
                'G_state_dict': netG.module.state_dict(),
                'D_state_dict': netD.module.state_dict(),
                'i_batch': i_batch,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, path_to_backup)
        # out = (y_hat[0]*255).transpose(0,2)
        # torchvision.utils.save_image(x, 'recent_backup_x.png', nrow=len(x), normalize=True, range=(0, 1))
        # torchvision.utils.save_image(y, 'recent_backup_y.png', nrow=len(y), normalize=True, range=(0, 1))
        # torchvision.utils.save_image(l_y.unsqueeze(1).repeat(1,3,1,1), 'recent_backup_l_y.png', nrow=len(l_y), normalize=True, range=(0, 1))
        torchvision.utils.save_image(x_fake[:8], 'recent_backup_train_gan.png', nrow=8, normalize=True, range=(0, 1))
        print('...Done saving latest')
