#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
pjoin = os.path.join
import numpy as np
import math
import sys
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet
from my_utils import LogPrint, set_up_dir, get_CodeID, feat_visualize, check_path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/home4/wanghuan/Projects/20180918_KD_for_NST/TaskAgnosticDeepCompression/Bin_CIFAR10/data_MNIST')
parser.add_argument('--teacher_dir', type=str, default='MNIST_model/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='MNIST_model/')
parser.add_argument('-p', '--project_name', type=str, default="")
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--CodeID', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--use_sign', action="store_true")
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--plot_train_feat', action="store_true")
opt = parser.parse_args()

# set up log dirs
TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(opt.project_name, opt.resume, opt.debug)
opt.output_dir = weights_path
logprint = LogPrint(log, ExpID)
opt.ExpID = ExpID
opt.CodeID = get_CodeID()
logprint(opt.__dict__)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True 

accr = 0
accr_best = 0

class LeNet5_2neurons(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_2neurons, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,   2)
    self.fc6 = nn.Linear(  2,  10)
    self.relu = nn.ReLU(inplace=True)
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
        param.requires_grad = False
   
  def forward_2neurons(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        if opt.mode == "original":
          self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))
        elif opt.mode == "ours":
          self.l1 = nn.Sequential(nn.Linear(opt.latent_dim+10, 128*self.init_size**2)) # 2019/10/08, test my losses
        else:
          raise NotImplementedError
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
        
generator = Generator().cuda()

teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

if opt.dataset == 'MNIST':    
    # Configure data loader   
    net = LeNet5Half().cuda()
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))           
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=1, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

if opt.dataset != 'MNIST':  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10': 
        net = resnet.ResNet18().cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR10(opt.data,
                          train=False,
                          transform=transform_test)
    if opt.dataset == 'cifar100': 
        net = resnet.ResNet18(num_classes=100).cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR100(opt.data,
                          train=False,
                          transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4) # wh: why use different optimizers for non-MNIST?

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 160: # 0.4 * opt.n_epochs: # 800:
        lr = learing_rate
    elif epoch < 320: #0.8 * opt.n_epochs: # 1600:
        lr = 0.1 * learing_rate
    else:
        lr = 0.01 * learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# ----------
#  Training
# ----------
if opt.dataset == "MNIST":
  pretrained = "/home4/wanghuan/Pro*/20180918*/Task*2/AgnosticMC/Bin_CIFAR10/train*/trained_weights_lenet5_2neurons/w*/*E21S0*.pth"
  pretrained = check_path(pretrained)
  embed_net = LeNet5_2neurons(pretrained).eval().cuda()
  fig_train = plt.figure(); ax_train = fig_train.add_subplot(111)

one_hot = OneHotCategorical(torch.Tensor([0.1] * 10))
batches_done = 0
for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i in range(120):
        net.train()
        if opt.mode == "original":
          z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
          optimizer_G.zero_grad()
          optimizer_S.zero_grad()
          gen_imgs = generator(z)
          outputs_T, features_T = teacher(gen_imgs, out_feature=True)
          
          # --- 2019/10/12: visualize
          label = outputs_T.argmax(dim=1); if_right = torch.ones_like(label)
          if opt.plot_train_feat and i % 10 == 0:
            feat = embed_net.forward_2neurons(gen_imgs)
            ax_train = feat_visualize(ax_train, feat.data.cpu().numpy(), label.data.cpu().numpy(), if_right.data.cpu().numpy())
            if i % 100 == 0:
              save_train_feat_path = pjoin(rec_img_path, "%s_E%sS%s_feat-visualization-train.jpg" % (ExpID, epoch, i))
              ax_train.set_xlim([-20, 200])
              ax_train.set_ylim([-20, 200])
              fig_train.savefig(save_train_feat_path, dpi=400)
              fig_train = plt.figure(); ax_train = fig_train.add_subplot(111)
          # ---
          
          pred = outputs_T.data.max(1)[1]
          loss_activation = -features_T.abs().mean()
          loss_one_hot = criterion(outputs_T,pred)
          softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
          loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
          loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
          loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
          loss += loss_kd
          loss.backward()
          optimizer_G.step()
          optimizer_S.step()
        elif opt.mode == "ours":
          # --- 2019/10/08: test my losses
          half_bs = int(opt.batch_size / 2)
          noise1 = torch.randn(half_bs, opt.latent_dim).cuda()
          noise2 = torch.randn(half_bs, opt.latent_dim).cuda()
          noise_concat = torch.cat([noise1, noise2], dim=0)
          onehot_label = one_hot.sample_n(half_bs).view([half_bs, 10]).cuda()
          pseudo_label_concat = torch.cat([onehot_label, onehot_label], dim=0)
          label = pseudo_label_concat.argmax(dim=1)
          x = torch.cat([noise_concat, pseudo_label_concat], dim=1)
          
          gen_imgs = generator(x)
          outputs_T, features_T = teacher(gen_imgs, out_feature=1)
          embed_1, embed_2 = torch.split(features_T, half_bs, dim=0)
          loss_one_hot = criterion(outputs_T, label) ## loss 1
          x_cos = torch.mean(F.cosine_similarity(noise1, noise2))
          y_cos = torch.mean(F.cosine_similarity(embed_1, embed_2))
          if opt.use_sign:
            loss_activation = y_cos / x_cos * torch.sign(x_cos).detach() ## loss 2
          else:
            loss_activation = y_cos / x_cos
          outputs_S = net(gen_imgs.detach())
          loss_kd = kdloss(outputs_S, outputs_T.detach()) ## loss 3
          loss = loss_one_hot * opt.oh + loss_activation * opt.a + loss_kd
          optimizer_G.zero_grad()
          optimizer_S.zero_grad()
          loss.backward()
          optimizer_G.step()
          optimizer_S.step()
          loss_information_entropy = torch.zeros(1)
          # ---
        
        if i == 1:
            logprint("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
            
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    logprint('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    if accr > accr_best:
        torch.save(net, opt.output_dir + 'student')
        accr_best = accr
