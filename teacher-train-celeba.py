#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from lenet import LeNet5
import resnet
import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
from my_utils import LogPrint, set_up_dir, get_CodeID
from model import AlexNet
from data_loader import CelebA
import math
import time

parser = argparse.ArgumentParser(description='train-teacher-network')
parser.add_argument('--CelebA_attr_file', type=str, default="../../Dataset/CelebA/Anno/list_attr_celeba.txt")
parser.add_argument('-p', '--project_name', type=str, default='')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--CodeID', type=str, default='')
parser.add_argument('--debug', action="store_true")
parser.add_argument('-b', '--batchsize', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()
args.data_CelebA_train = "../../Dataset/CelebA/Img/train/"
args.data_CelebA_test = "../../Dataset/CelebA/Img/test/"
num_attributes = 40

# set up log dirs
TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(args.project_name, args.resume, args.debug)
args.output_dir = weights_path
logprint = LogPrint(log, ExpID)
args.ExpID = ExpID
args.CodeID = get_CodeID()
logprint(args.__dict__)
os.makedirs(args.output_dir, exist_ok=True)

# set up data
# much is referred to https://github.com/d-li14/face-attribute-prediction/blob/840e59aee39df1c21d149bc0cd221b914a700872/main.py
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
net = AlexNet().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) # according to DAFL, use Adam with 1e-4 lr

def train(epoch):
  net.train()
  losses = [AverageMeter() for _ in range(num_attributes)]
  top1 = [AverageMeter() for _ in range(num_attributes)]
  
  for i, (input, target) in enumerate(data_train_loader):
      input = input.cuda(); target = target.cuda()
      output = net(input)
      
      loss = []; accu = []
      for j in range(len(output)):
        loss.append(criterion(output[j], target[:, j]))
        accu.append(accuracy(output[j], target[:, j], topk=(1,)))
        losses[j].update(loss[j].item(), input.size(0))
        top1[j].update(accu[j][0].item(), input.size(0))
      losses_avg = [losses[k].avg for k in range(len(losses))]
      top1_avg = [top1[k].avg for k in range(len(top1))]
      loss_avg = sum(losses_avg) / len(losses_avg)
      accu_avg = sum(top1_avg) / len(top1_avg)
      
      optimizer.zero_grad()
      loss_sum = sum(loss)
      loss_sum.backward()
      optimizer.step()
      
      if i % 10 == 0:
        logprint('Train - E%dS%d/%d, Loss: %f' % (epoch, i, 
            math.ceil(len(data_train) / args.batchsize), loss_sum.data.item()))
        
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

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                  batch=i+1,
                  size=len(test_loader),
                  data=data_time.avg,
                  bt=batch_time.avg,
                  loss=loss_avg,
                  top1=accu_avg,
                  )
      logprint(suffix)
  return accu_avg
 
def train_and_test(epoch):
  train(epoch)
  acc = test(data_test_loader, net, criterion)
  return acc
 
def main():
  test(data_test_loader, net, criterion)
  epoch = 10 # it is very easy to converge, so 10 epochs is enough.
  max_acc = 0
  for e in range(epoch):
    acc = train_and_test(e)
    if acc > max_acc:
      max_acc = acc
      torch.save(net, args.output_dir + '/teacher')

if __name__ == '__main__':
    main()