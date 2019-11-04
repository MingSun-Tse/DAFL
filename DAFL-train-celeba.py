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
from model import AlexNet_half

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100', 'celeba'])
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
parser.add_argument('--oscill_thre', type=float, default=1000) # deprecated. used to be 2e-2
parser.add_argument('--n_try', type=int, default=5)
parser.add_argument('--multiplier', type=float, default=2)
parser.add_argument('--temp', type=float, default=0.2)
parser.add_argument('--ema', type=float, default=0.9)
parser.add_argument('--label_oh', action="store_true")
parser.add_argument('--use_detach_for_my_oh', action="store_true")
opt = parser.parse_args()
opt.oscill_thre *= (math.log10(math.e) * opt.num_class)

if opt.dataset != "MNIST":
  opt.channels = 3
if opt.dataset == "cifar100":
  opt.num_class = 100
if opt.dataset == "imagenet":
  opt.num_class = 1000
if opt.dataset == "celeba":
  opt.num_class = 2

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
opt.data = ""
opt.teacher_dir = ""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
data_train = CelebA(args.data_CelebA_train, args.CelebA_attr_file, transform=transform_train)
data_test  = CelebA(args.data_CelebA_test,  args.CelebA_attr_file, transform=transform_test)
data_train_loader = DataLoader(data_train, batch_size=args.batchsize, shuffle=True, num_workers=0)
data_test_loader  = DataLoader(data_test,  batch_size=args.batchsize, num_workers=0)

# set up model
teacher = torch.load(opt.teacher_dir + '/teacher').cuda()
teacher.eval()
teacher = nn.DataParallel(teacher)
generator = Generator().cuda()
generator = nn.DataParallel(generator)
net = AlexNet_half().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer_S = torch.optim.Adam(net.parameters(), lr=args.lr_S) # according to DAFL, use Adam with 1e-4 lr
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
criterion = torch.nn.CrossEntropyLoss().cuda()

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl
    
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)

      _, pred = output.topk(maxk, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))

      res = []
      for k in topk:
          correct_k = correct[:k].view(-1).float().sum(0)
          res.append(correct_k.mul_(100.0 / batch_size))
      return res
        
class AverageMeter(object):
  """Computes and stores the average and current value
     Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
  """
  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

def test(test_loader, model, criterion):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = [AverageMeter() for _ in range(num_attributes)]
  top1 = [AverageMeter() for _ in range(num_attributes)]

  model.eval().cuda()
  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
      # measure data loading time
      data_time.update(time.time() - end)
      input = input.cuda()
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      # measure accuracy and record loss
      loss = []; accu = []
      for j in range(len(output)): # the j-th attr
        loss.append(criterion(output[j], target[:, j])) # output[j] shape: batch_size x 2
        accu.append(accuracy(output[j], target[:, j], topk=(1,)))

        losses[j].update(loss[j].item(), input.size(0))
        top1[j].update(accu[j][0].item(), input.size(0))
      
      # update loss and acc
      losses_avg = [losses[k].avg for k in range(len(losses))] # losses for 40 attrs
      top1_avg = [top1[k].avg for k in range(len(top1))]
      loss_avg = sum(losses_avg) / len(losses_avg)
      accu_avg = sum(top1_avg) / len(top1_avg)

  return accu_avg
    
# Optimizers
# optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4) # wh: why use different optimizers for non-MNIST?

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
  ema_G = EMA(opt.ema)
  for name, param in generator.named_parameters():
    if param.requires_grad:
      ema_G.register(name, param.data)

# ----------
#  Training
# ----------
acc = 0; max_acc = 0
batches_done = 0
sample_prob = torch.ones(opt.num_class) / opt.num_class
num_sample_per_class = [0] * opt.num_class
history_acc_S = [0] * opt.num_class
history_kld_S = [0] * opt.num_class
history_prob_var = [0] * opt.num_class
history_ie = 0
num_attributes = 40
for epoch in range(opt.n_epochs):
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for step in range(opt.num_iter_per_epoch):
        net.train()
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
          outputs_T, features_T = teacher(gen_imgs, embed=True)
          
          # oh loss
          loss_one_hot = torch.zeros(1)
          if opt.oh:
            for attr in range(num_attributes):
              prob = F.softmax(outputs_T[attr], dim=1)
              enhanced_prob = F.softmax(outputs_T[attr] / opt.temp, dim=1)
              loss_one_hot = F.kl_div(prob.log(), enhanced_prob.data) * opt.num_class
              loss_G += loss_one_hot * opt.oh

          # ie loss
          update_dist_cond = epoch >= 2
          loss_information_entropy = torch.zeros(1)
          if opt.ie:
            # # print to check
            # if step % opt.show_interval == 0 and gi == opt.n_G_update-1:
              # logtmp1 = ""; logtmp2 = ""
              # for c in range(opt.num_class):
                # logtmp1 += "%.4f  " % history_acc_S[c]
                # # logtmp2 += "%.4f  " % history_kld_S[c]
              # logprint(logtmp1 + "train history_acc_S (E%dS%d) ave = %.4f" % (epoch, step, np.mean(history_acc_S)))
              # # logprint(logtmp2 + "train kld_T_S       (E%dS%d) ave = %.4f" % (epoch, step, np.mean(history_kld_S)))
            for attr in range(num_attributes):
              actual_dist = F.softmax(outputs_T[attr], dim=1).mean(dim=0)
              if step % opt.update_dist_interval == 0:
                if opt.uniform_target_dist or (not update_dist_cond):
                  expect_dist = torch.ones(opt.num_class).cuda() / opt.num_class
                  temp = 0
                else:
                  temp = (max(history_acc_S) - min(history_acc_S)) / math.log(opt.multiplier)
                  expect_dist = F.softmax(-torch.from_numpy(np.array(history_acc_S)) / temp, dim=0).cuda().float()
              loss_information_entropy = F.kl_div(expect_dist.log().detach(), actual_dist) * opt.num_class * math.log10(math.e)
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
              loss_G += loss_information_entropy * ie_lw
            
          # cos loss
          update_coslw_cond = np.mean(history_acc_S) > opt.base_acc
          if opt.a and update_coslw_cond:
            embed_1, embed_2 = torch.split(features_T, half_bs, dim=0)
            x_cos = F.cosine_similarity(noise_1, noise_2)
            y_cos = F.cosine_similarity(embed_1, embed_2)
            loss_activation = y_cos / torch.abs(x_cos) if opt.use_sign else y_cos / x_cos
            loss_activation = loss_activation.mean()
            loss_G += loss_activation * opt.a
          else:
            loss_activation = torch.zeros(1).cuda()
          
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
        
        
        if step % opt.show_interval == 0:
            logprint("E%dS%d/%d: [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, step, opt.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
        
        if step % opt.test_interval == 0:
          acc = test()
          logprint("=" * (int(ExpID[-1])+1) + '> E%dS%d: Accuracy: %f' % (epoch, step, acc))
          if acc > max_acc:
              torch.save(net, opt.output_dir + '/student')
              max_acc = acc
