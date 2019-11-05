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
import time
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
from model import AlexNet_half, DCGAN_Generator
from data_loader import CelebA

parser = argparse.ArgumentParser()
parser.add_argument('--CelebA_attr_file', type=str, default="../../Dataset/CelebA/Anno/list_attr_celeba.txt")
parser.add_argument('--teacher_dir', type=str, default='MNIST_model/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--num_iter_per_epoch', type=int, default=120)
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--lw_adv', type=float, default=0)
parser.add_argument('-p', '--project_name', type=str, default="")
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--CodeID', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--use_sign', action="store_true")
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--test_interval', type=int, default=120)
parser.add_argument('--show_interval', type=int, default=10)
parser.add_argument('--update_dist_interval', type=int, default=60)
parser.add_argument('--momentum_cnt', type=float, default=0.9)
parser.add_argument('--uniform_target_dist', action="store_true")
parser.add_argument('--n_G_update', type=int, default=1)
parser.add_argument('--n_S_update', type=int, default=1)
parser.add_argument('--base_acc', type=float, default=0.4)
parser.add_argument('--multiplier', type=float, default=2)
parser.add_argument('--temp', type=float, default=0.4)
parser.add_argument('--ema', type=float, default=0.9)
parser.add_argument('--noise_as_input', action="store_true")
parser.add_argument('--lr_S', type=float, default=1e-2)
opt = parser.parse_args()

# set up data
num_attributes = 40
opt.num_class = 2
opt.teacher_dir = "Experiments/SERVER5-20191104-003809_CelebA-Teacher/weights/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize])
opt.data_CelebA_test = "../../Dataset/CelebA/Img/test/"
data_test = CelebA(opt.data_CelebA_test,  opt.CelebA_attr_file, transform=transform_test)
data_test_loader = DataLoader(data_test, batch_size=256, num_workers=0)

# set up model
teacher = torch.load(opt.teacher_dir + '/teacher').cuda()
teacher.eval()
teacher = nn.DataParallel(teacher)
generator = DCGAN_Generator(opt.latent_dim).cuda()
generator = nn.DataParallel(generator)
net = AlexNet_half().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer_S = torch.optim.Adam(net.parameters(), lr=1e-4) # according to DAFL, use Adam with 1e-4 lr
#optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

def accuracy(output, target, topk=(1,)):
  """ Computes the precision@k for the specified values of k.
      output: bs x 2
      target: bs x 1
  """
  with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)
      _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) # return the largest k elements of the tensor along dim = 1
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
    top1_avg = [top1[k].avg for k in range(len(top1))] # the acc per attr
    loss_avg = sum(losses_avg) / len(losses_avg)
    accu_avg = sum(top1_avg) / len(top1_avg)
    logtmp = "\n"
    for attr in range(num_attributes):
      logtmp += "%.2f " % top1_avg[attr]
    logprint(logtmp)
  return accu_avg

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

  
def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch < 1: 
        lr = learning_rate
    elif epoch < 2:
        lr = 0.1 * learning_rate
    elif epoch < 3:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# ----------
#  Training
# ----------
acc = 0; max_acc = 0
batches_done = 0
sample_prob = torch.ones(opt.num_class) / opt.num_class
num_sample_per_class = [0] * opt.num_class
history_acc_S = np.zeros([num_attributes, opt.num_class])
history_kld_S = np.zeros([num_attributes, opt.num_class])
# teacher_acc = test(data_test_loader, teacher, criterion) # check the acc of the teacher
# logprint("teacher's acc = %.4f" % teacher_acc)
for epoch in range(opt.n_epochs):
  for step in range(opt.num_iter_per_epoch):
    # test
    if step % opt.test_interval == 0 and epoch + step != 0:
      acc = test(data_test_loader, net, criterion)
      logprint("=" * (int(ExpID[-1])+1) + '> E%dS%d: Accuracy: %f' % (epoch, step, acc))
      if acc > max_acc:
        torch.save(generator, opt.output_dir + '/generator')
        torch.save(net, opt.output_dir + '/student')
        max_acc = acc
    
    # adjust_learning_rate(optimizer_S, epoch, opt.lr_S)
    # set up input noise
    net.train()
    half_bs = int(opt.batch_size / 2)
    noise_1 = torch.randn(half_bs, opt.latent_dim).cuda()
    noise_2 = torch.randn(half_bs, opt.latent_dim).cuda()
    x = torch.cat([noise_1, noise_2], dim=0)
    x = x.view(x.size(0), x.size(1), 1, 1)
    
    # update G
    if not opt.noise_as_input:
      gi = 0; n_stuck_in_loop = {}
      while gi < opt.n_G_update:
        loss_G = torch.zeros(1).cuda()
        gen_imgs = generator(x)
        outputs_T, features_T = teacher(gen_imgs, embed=True)
        outputs_S             = net(gen_imgs)
        
        # oh loss and ie loss
        for attr in range(num_attributes):
          # if attr != 2: continue
          outT = outputs_T[attr] # outputs_T: 40 x (bs x 2)
          outS = outputs_S[attr]
          
          # oh loss
          prob = F.softmax(outT, dim=1)
          enhanced_prob = F.softmax(outT / opt.temp, dim=1)
          loss_one_hot = F.kl_div(prob.log(), enhanced_prob.data) * opt.num_class
          loss_G += loss_one_hot * opt.oh
          
          var_loss = torch.var(prob, dim=0).mean()
          loss_G += -var_loss * 1000
          
          # ie loss
          if opt.uniform_target_dist:
            expect_dist = torch.ones(opt.num_class).cuda() / opt.num_class
            temp = 0
          else:
            kld = F.kl_div(F.log_softmax(outS, dim=1), F.softmax(outT, dim=1), reduction="none")
            kld = kld.mean(dim=0)
            temp = (max(kld) - min(kld)) / math.log(opt.multiplier)
            expect_dist = F.softmax(kld / temp.data, dim=0).cuda()
          actual_dist = F.softmax(outT, dim=1).mean(dim=0)
          aa1 = F.kl_div(expect_dist.log(), actual_dist)
          aa2 = F.kl_div(actual_dist.log(), expect_dist)
          loss_information_entropy = (aa1 + aa2) * opt.num_class * math.log10(math.e)
          # print(expect_dist, actual_dist)
          loss_G += loss_information_entropy * opt.ie
          
        # cos loss
        # loss_activation = -features_T.abs().mean()
        # loss_G += loss_activation * opt.a
        update_coslw_cond = 1
        loss_activation = torch.zeros(1).cuda()
        if update_coslw_cond:
          embed_1, embed_2 = torch.split(features_T, half_bs, dim=0)
          x_cos = F.cosine_similarity(noise_1, noise_2)
          y_cos = F.cosine_similarity(embed_1, embed_2)
          loss_activation = y_cos / torch.abs(x_cos) if opt.use_sign else y_cos / x_cos
          loss_activation = loss_activation.mean()
          loss_G += loss_activation * opt.a
        
        # 2019/10/21 EMA to avoid collpase
        optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()
        for name, param in generator.named_parameters():
          if param.requires_grad:
            param.data = ema_G(name, param.data)
        gi += 1
    else:
      gen_imgs = torch.randn([opt.batch_size, 3, 224, 224]).cuda()
      outputs_T = teacher(gen_imgs)
      loss_one_hot = loss_information_entropy = loss_activation = torch.zeros(1)
      temp = 0
    
    # update S
    for si in range(opt.n_S_update):
      outputs_S = net(gen_imgs.detach())
      loss_kd = 0
      for attr in range(num_attributes):
        outT = outputs_T[attr]
        outS = outputs_S[attr]
        loss_kd += kdloss(outS, outT.detach())
      optimizer_S.zero_grad(); loss_kd.backward(); optimizer_S.step()
    
      # get per-class status of the Student
      if step % opt.show_interval == 0:
        for attr in range(num_attributes):
          # if attr != 2: continue
          outT = outputs_T[attr]
          outS = outputs_S[attr]
          acc = [0] * opt.num_class; cnt = [1] * opt.num_class; kld = [0] * opt.num_class
          
          # check the label of T
          prob = F.softmax(outT, dim=1)
          labelT = outT.argmax(dim=1).data.cpu().numpy()
          print("attr %2d: sample ratio of 0 = %.2f, 1 = %.2f" % (attr, sum(labelT==0)/opt.batch_size, sum(labelT==1)/opt.batch_size))
          
          for oS, oT in zip(outS, outT):
            lT = oT.argmax(); lS = oS.argmax()
            acc[lT] += int(lT == lS)
            cnt[lT] += 1
            kld[lT] += F.kl_div(F.log_softmax(oS, dim=0), F.softmax(oT, dim=0)).item()
          for c in range(opt.num_class):
            current_acc = acc[c] * 1.0 / cnt[c]
            current_kld = kld[c] * 1.0 / cnt[c]
            history_acc_S[attr, c] = history_acc_S[attr, c] * opt.momentum_cnt + current_acc * (1-opt.momentum_cnt) if history_acc_S[attr, c] else current_acc
            history_kld_S[attr, c] = history_kld_S[attr, c] * opt.momentum_cnt + current_kld * (1-opt.momentum_cnt) if history_kld_S[attr, c] else current_kld
        
        logtmp1 = "\n"; logtmp2 = "\n"
        for attr in range(num_attributes):
          logtmp1 += "%.2f " % history_acc_S[attr, 1]
          logtmp2 += "%.2f " % history_kld_S[attr, 1]
        logprint(logtmp1)
        logprint(logtmp2)
    
    if step % opt.show_interval == 0:
      logprint("E%dS%d/%d: [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f] temp: %.2f" % (epoch, step, opt.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item(), temp))