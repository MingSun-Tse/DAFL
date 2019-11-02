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
from my_utils import LogPrint, set_up_dir, get_CodeID, feat_visualize, check_path, EMA

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='../20180918_KD_for_NST/TaskAgnosticDeepCompression/Bin_CIFAR10/data_MNIST')
parser.add_argument('--teacher_dir', type=str, default='MNIST_model/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_iter_per_epoch', type=int, default=120)
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--lw_norm', type=float, default=0)
parser.add_argument('--lw_adv', type=float, default=0)
parser.add_argument('--lw_prob_var', type=float, default=0)
parser.add_argument('--output_dir', type=str, default='MNIST_model/')
parser.add_argument('-p', '--project_name', type=str, default="")
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--CodeID', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--use_sign', action="store_true")
parser.add_argument('--use_condition', action="store_true")
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--plot_train_feat', action="store_true")
parser.add_argument('--which_lenet', type=str, default="")
parser.add_argument('--adjust_sampler', action="store_true")
parser.add_argument('--test_interval', type=int, default=120)
parser.add_argument('--show_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--update_dist_interval', type=int, default=60)
parser.add_argument('--momentum_cnt', type=float, default=0.9)
parser.add_argument('--uniform_target_dist', action="store_true")
parser.add_argument('--n_G_update', type=int, default=1)
parser.add_argument('--n_S_update', type=int, default=1)
parser.add_argument('--base_acc', type=float, default=0.4)
parser.add_argument('--oscill_thre', type=float, default=2e-2)
parser.add_argument('--n_try', type=int, default=5)
parser.add_argument('--multiplier', type=float, default=2)
opt = parser.parse_args()
opt.oscill_thre *= (math.log10(math.e) * opt.num_class)

if opt.dataset != "MNIST":
  opt.channels = 3
if opt.dataset == "cifar100":
  opt.num_class = 100
if opt.dataset == "imagenet":
  opt.num_class = 1000
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
  
  def forward(self, y, out_feature=False):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    feat = y.view(y.size(0), -1)
    y = self.relu(self.fc3(feat))
    y = self.relu(self.fc4(y))
    y = self.relu(self.fc5(y))
    y = self.fc6(y)
    if out_feature:
      return y, feat
    else:
      return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        if opt.mode == "original":
          self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))
        elif opt.mode == "ours":
          self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))
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
        
# set up data
if opt.dataset == "cifar10":
  opt.data= "../20180918_KD_for_NST/TaskAgnosticDeepCompression2/AgnosticMC/Bin_CIFAR10/data_CIFAR10"
  opt.teacher_dir = "CIFAR10_model/"
if opt.dataset == "cifar100":
  opt.data = "../20180918_KD_for_NST/TaskAgnosticDeepCompression2/AgnosticMC/Bin_CIFAR10/data_CIFAR100"
  opt.teacher_dir = "Experiments/SERVER218-20191022-094454_teacher-cifar100/weights/"
  
# set up model
teacher = torch.load(opt.teacher_dir + '/teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
if opt.dataset == "MNIST" and "_2neurons" in opt.which_lenet:
  if opt.which_lenet == "_2neurons1":
    pretrained = "../20180918*/Task*2/AgnosticMC/Bin_CIFAR10/train*/trained_weights_lenet5_2neurons/w*/*E21S0*.pth"
  elif opt.which_lenet == "_2neurons2":
    pretrained = "../20180918*/Task*2/AgnosticMC/Bin_CIFAR10/train*/trained_weights_lenet5_2neurons_2/w*/*E23S0*.pth"
  pretrained = check_path(pretrained)
  teacher = LeNet5_2neurons(pretrained).eval().cuda()
if opt.dataset == "MNIST": # set up embed net
  pretrained = "../20180918*/Task*2/AgnosticMC/Bin_CIFAR10/train*/trained_weights_lenet5_2neurons/w*/*E21S0*.pth"
  pretrained = check_path(pretrained)
  embed_net = LeNet5_2neurons(pretrained).eval().cuda()
  fig_train = plt.figure(); ax_train = fig_train.add_subplot(111)
teacher = nn.DataParallel(teacher)
generator = Generator().cuda()
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

def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch < 160: # 0.4 * opt.n_epochs: # 800:
        lr = learning_rate
    elif epoch < 320: #0.8 * opt.n_epochs: # 1600:
        lr = 0.1 * learning_rate
    elif epoch < 480:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# set up log dirs
TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(opt.project_name, opt.resume, opt.debug)
opt.output_dir = weights_path
logprint = LogPrint(log, ExpID)
opt.ExpID = ExpID
opt.CodeID = get_CodeID()
logprint(opt.__dict__)

if opt.mode == "ours":
  ema_G = EMA(0.9)
  for name, param in generator.named_parameters():
    if param.requires_grad:
      ema_G.register(name, param.data)

# ----------
#  Training
# ----------
batches_done = 0
sample_prob = torch.ones(opt.num_class) / opt.num_class
num_sample_per_class = [0] * opt.num_class
history_acc_S = [0] * opt.num_class
history_kld_S = [0] * opt.num_class
history_prob_var = [0] * opt.num_class
history_ie = 0
for epoch in range(opt.n_epochs):
    # total_correct = 0
    # avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for step in range(opt.num_iter_per_epoch):
        net.train()
        if opt.mode == "original":
          z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
          optimizer_G.zero_grad()
          optimizer_S.zero_grad()
          gen_imgs = generator(z)
          outputs_T, features_T = teacher(gen_imgs, out_feature=True)
          
          # -2019/10/12: visualize
          label = outputs_T.argmax(dim=1); if_right = torch.ones_like(label)
          if opt.plot_train_feat and step % 10 == 0:
            feat = embed_net.forward_2neurons(gen_imgs)
            ax_train = feat_visualize(ax_train, feat.data.cpu().numpy(), label.data.cpu().numpy(), if_right.data.cpu().numpy())
            if step % opt.save_interval == 0:
              save_train_feat_path = pjoin(rec_img_path, "%s_E%sS%s_feat-visualization-train.jpg" % (ExpID, epoch, step))
              ax_train.set_xlim([-20, 200])
              ax_train.set_ylim([-20, 200])
              fig_train.savefig(save_train_feat_path, dpi=400)
              fig_train = plt.figure(); ax_train = fig_train.add_subplot(111)
          # ---
          
          pred = outputs_T.data.max(1)[1]
          loss_activation = -features_T.abs().mean()
          loss_one_hot = criterion(outputs_T,pred)
          softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
          # loss_information_entropy1 = (softmax_o_T * torch.log(softmax_o_T)).sum()
          expect_dist = torch.ones(opt.num_class).cuda() / opt.num_class
          loss_information_entropy = F.kl_div(expect_dist.log(), softmax_o_T) * opt.num_class * math.log10(math.e)
          # print(loss_information_entropy1.item(), loss_information_entropy.item(), loss_information_entropy1.item()+math.log(opt.num_class))
          loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
          loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
          loss += loss_kd
          loss.backward()
          optimizer_G.step()
          optimizer_S.step()

          # analyze label_T
          logtmp = ""
          for c in range(opt.num_class):
            tmp = sum(label.cpu().data.numpy() == c)
            num_sample_per_class[c] = num_sample_per_class[c] * opt.momentum_cnt + tmp * (1-opt.momentum_cnt) if num_sample_per_class[c] else tmp
            logtmp += "%.4f  " % (num_sample_per_class[c] / opt.batch_size)
          if step % opt.show_interval == 0:
            logprint(logtmp + ("real class ratio (E%dS%d)" % (epoch, step)))
          
        elif opt.mode == "ours":
          half_bs = int(opt.batch_size / 2)
          noise_1 = torch.randn(half_bs, opt.latent_dim).cuda()
          noise_2 = torch.randn(half_bs, opt.latent_dim).cuda()
          x = torch.cat([noise_1, noise_2], dim=0)
          
          # update G
          gi = 0
          n_stuck_in_loop = {}
          while gi < opt.n_G_update:
            loss_G = torch.zeros(1).cuda()
            gen_imgs = generator(x)
            outputs_T, features_T = teacher(gen_imgs, out_feature=1)
            label_T = outputs_T.argmax(dim=1)

            # one hot loss
            if opt.oh:
              loss_one_hot = criterion(outputs_T, label)
              loss_G += loss_one_hot * opt.oh
            else:
              loss_one_hot = torch.zeros(1)
            
            # prob variance per class loss
            if opt.lw_prob_var:
              var = torch.var(F.softmax(outputs_T, dim=1), dim=0)
              loss_G += var.mean() * opt.lw_prob_var
              # logtmp = ""
              # for c in range(opt.num_class):
                # history_prob_var[c] = history_prob_var[c] * opt.momentum_cnt + var[c] * (1-opt.momentum_cnt) \
                    # if history_prob_var[c] else var[c]
                # logtmp += "%.4f  " % history_prob_var[c].item()
            
            # ie loss
            update_dist_cond = np.mean(history_acc_S) > opt.base_acc
            loss_information_entropy = torch.zeros(1)
            if opt.ie:
              # print to check
              if step % opt.show_interval == 0 and gi == opt.n_G_update-1:
                logtmp1 = ""; logtmp2 = ""
                for c in range(opt.num_class):
                  logtmp1 += "%.4f  " % history_acc_S[c]
                  # logtmp2 += "%.4f  " % history_kld_S[c]
                logprint(logtmp1 + "train history_acc_S (E%dS%d) ave = %.4f" % (epoch, step, np.mean(history_acc_S)))
                # logprint(logtmp2 + "train kld_T_S       (E%dS%d) ave = %.4f" % (epoch, step, np.mean(history_kld_S)))
              '''
              # scheme 1: use kld to adjust class ratio
              kl = F.kl_div(F.log_softmax(outputs_S, dim=1), F.softmax(outputs_T, dim=1), reduction="none")
              kl = kl.mean(dim=0)
              if opt.uniform_target_dist or (not update_dist_cond):
                expect_dist = torch.ones(opt.num_class).cuda() / opt.num_class
                temp = 0
              else:
                temp = (kl.max() - kl.min()) / math.log(opt.multiplier)
                expect_dist = F.softmax(kl / temp, dim=0)
              actual_dist = F.softmax(outputs_T, dim=1).mean(dim=0)
              loss_information_entropy = F.kl_div(actual_dist.log(), expect_dist)
              '''
              # scheme 2: use history acc of Student to adjust class ratio
              actual_dist = F.softmax(outputs_T, dim=1).mean(dim=0)
              if step % opt.update_dist_interval == 0:
                if opt.uniform_target_dist or (not update_dist_cond):
                  expect_dist = torch.ones(opt.num_class).cuda() / opt.num_class
                  temp = 0
                else:
                  temp = (max(history_acc_S) - min(history_acc_S)) / math.log(opt.multiplier)
                  expect_dist = F.softmax(-torch.from_numpy(np.array(history_acc_S)) / temp, dim=0).cuda().float()
              loss_information_entropy = F.kl_div(actual_dist.log(), expect_dist.detach()) * opt.num_class * math.log10(math.e)
              history_ie = opt.momentum_cnt * history_ie + (1-opt.momentum_cnt) * loss_information_entropy.item()
              
              # print to check
              if step % opt.show_interval == 0 and gi == opt.n_G_update-1:
                logtmp1 = ""; logtmp2 = ""
                for c in range(opt.num_class):
                  logtmp1 += "%.4f  " % expect_dist[c]
                  logtmp2 += "%.4f  " % actual_dist[c]
                logprint(logtmp1 + ("expected class ratio (E%dS%d) ie: %.4f histie: %.4f temp: %.2f" % (epoch, step, loss_information_entropy, history_ie, temp)))
                logprint(logtmp2 + ("real     class ratio (E%dS%d)" % (epoch, step)))
              
              # oscillation check
              ie_lw = opt.ie
              if epoch > 1 and history_ie > opt.oscill_thre:
              # the first epoch does not check oscillation because it is normal to oscillate at the beginning
                current_step = str(epoch * opt.num_iter_per_epoch + step)
                if current_step in n_stuck_in_loop:
                  n_stuck_in_loop[current_step] += 1
                else:
                  n_stuck_in_loop[current_step] = 1
                if n_stuck_in_loop[current_step] > opt.n_try:
                  logprint("cannot escape from this bad oscillation, stop trying, just skip it")
                  loss_G.backward() # to clear gradients
                  break
                logprint("some bad oscillation happens, extend the G's training time to stable the class ratio (E%dS%d gi=%d) #%d" % 
                    (epoch, step, gi, n_stuck_in_loop[current_step]))
                ie_lw = opt.ie * 2
                gi -= 1 # the loop will not stop unless the class ratios are corrected
              loss_G += loss_information_entropy * ie_lw
              
            # cos loss
            update_coslw_cond = np.mean(history_acc_S) > opt.base_acc
            if update_coslw_cond:
              embed_1, embed_2 = torch.split(features_T, half_bs, dim=0)
              x_cos = F.cosine_similarity(noise_1, noise_2)
              y_cos = F.cosine_similarity(embed_1, embed_2)
              sign = (label_T[:half_bs] == label_T[half_bs:]).detach().float()
              loss_activation = y_cos / torch.abs(x_cos) * sign if opt.use_sign else y_cos / x_cos * sign
              loss_activation = loss_activation.sum()
              loss_activation /= sign.sum()
            else:
              loss_activation = torch.zeros(1).cuda()
            loss_G += loss_activation * opt.a
              
            
            # 2019/10/21 EMA to avoid collpase
            optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()
            for name, param in generator.named_parameters():
              if param.requires_grad:
                param.data = ema_G(name, param.data)
            gi += 1
          
          # update S
          for si in range(opt.n_S_update):
            outputs_S = net(gen_imgs.detach())
            pred = outputs_S.max(1)[1]
            if_right = pred.eq(label_T.view_as(pred))
            loss_kd = kdloss(outputs_S, outputs_T.detach())
            optimizer_S.zero_grad(); loss_kd.backward(); optimizer_S.step()
          
            # get per-class status of the Student
            acc = [0] * opt.num_class; cnt = [1] * opt.num_class; kld = [0] * opt.num_class
            for oS, oT in zip(outputs_S, outputs_T):
              lT = oT.argmax(); lS = oS.argmax()
              acc[lT] += int(lT == lS)
              cnt[lT] += 1
              kld[lT] += F.kl_div(F.log_softmax(oS, dim=0), F.softmax(oT, dim=0)).item()
            for c in range(opt.num_class):
              current_acc = acc[c] * 1.0 / cnt[c]
              current_kld = kld[c] * 1.0 / cnt[c]
              history_acc_S[c] = history_acc_S[c] * opt.momentum_cnt + current_acc * (1-opt.momentum_cnt) if history_acc_S[c] else current_acc
              history_kld_S[c] = history_kld_S[c] * opt.momentum_cnt + current_kld * (1-opt.momentum_cnt) if history_kld_S[c] else current_kld
          
          # visualize
          if_right = torch.ones_like(label_T)
          if opt.plot_train_feat and step % 10 == 0:
            feat = embed_net.forward_2neurons(gen_imgs)
            ax_train = feat_visualize(ax_train, feat.data.cpu().numpy(), label_T.data.cpu().numpy(), if_right.data.cpu().numpy())
            if step % opt.save_interval == 0:
              save_train_feat_path = pjoin(rec_img_path, "%s_E%sS%s_feat-visualization-train.jpg" % (ExpID, epoch, step))
              ax_train.set_xlim([-20, 200])
              ax_train.set_ylim([-20, 200])
              fig_train.savefig(save_train_feat_path, dpi=400)
              fig_train = plt.figure(); ax_train = fig_train.add_subplot(111)
          
        else:
          raise NotImplementedError
        
        if step % opt.show_interval == 0:
            logprint("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
        
        if step % opt.test_interval == 0:
          total_correct = 0
          avg_loss = 0
          with torch.no_grad():
              pred_total = []; label_total = []
              for _, (images, labels) in enumerate(data_test_loader):
                  images = images.cuda()
                  labels = labels.cuda()
                  net.eval()
                  output = net(images)
                  avg_loss += criterion(output, labels).sum()
                  pred = output.data.max(1)[1]
                  total_correct += pred.eq(labels.data.view_as(pred)).sum()
                  pred_total.extend(list(pred.data.cpu().numpy()))
                  label_total.extend(list(labels.data.cpu().numpy()))
          acc_test = [0] * opt.num_class; cnt_test = [0] * opt.num_class
          for p, l in zip(pred_total, label_total):
            acc_test[l] += int(p == l)
            cnt_test[l] += 1
          logtmp = ""
          for c in range(opt.num_class):
            acc_test[c] /= float(cnt_test[c])
            logtmp += "%.4f  " % acc_test[c]
          logprint(logtmp + ("test acc per class (E%dS%d)" % (epoch, step)))
    
          avg_loss /= len(data_test)
          logprint("=" * (int(ExpID[-1])+1) + '> E%dS%d: Test Avg. Loss: %f, Accuracy: %f' % (epoch, step, avg_loss.data.item(), 
              float(total_correct) / len(data_test)))
          accr = round(float(total_correct) / len(data_test), 4)
          if accr > accr_best:
              torch.save(net, opt.output_dir + '/student')
              accr_best = accr