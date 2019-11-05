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

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100', 'celeba',])
parser.add_argument('--data', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('-p', '--project_name', type=str, default='')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--CodeID', type=str, default='')
parser.add_argument('--debug', action="store_true")
parser.add_argument('--which_net', type=str, default="")
parser.add_argument('-b', '--batchsize', type=int, default=256)
args = parser.parse_args()

if args.dataset == "celeba":
  args.data_CelebA_train = "../../Dataset/CelebA/Img/train/"
  args.data_CelebA_test  = "../../Dataset/CelebA/Img/test/"
  args.CelebA_attr_file  = "../../Dataset/CelebA/Anno/list_attr_celeba.txt"

# set up log dirs
TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(args.project_name, args.resume, args.debug)
args.output_dir = weights_path
logprint = LogPrint(log, ExpID)
args.ExpID = ExpID
args.CodeID = get_CodeID()
logprint(args.__dict__)

os.makedirs(args.output_dir, exist_ok=True)

if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                       transform=transform_train)
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

    if args.which_net == "embed":
      net = resnet.ResNet34_2neurons().cuda()
    else:
      net = resnet.ResNet34().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                       transform=transform_train)
    data_test = CIFAR100(args.data,
                      train=False,
                      transform=transform_test)
                      
    data_train_loader = DataLoader(data_train, batch_size=args.batchsize, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=args.batchsize, num_workers=0)
    net = resnet.ResNet34(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'celeba':
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
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) # according to DAFL, use Adam with 1e-4 lr

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    if args.dataset not in ['MNIST', 'celeba']:
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    n_iter_per_epoch = math.ceil(len(data_train) / args.batchsize)
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images, embed=False)

        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i % 100 == 0:
            logprint('Train - E%dS%d/%d, Loss: %f' % (epoch, i, n_iter_per_epoch, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
def test():
  net.eval()
  total_correct = 0
  avg_loss = 0.0
  with torch.no_grad():
      for i, (images, labels) in enumerate(data_test_loader):
          images, labels = Variable(images).cuda(), Variable(labels).cuda()
          output = net(images)
          avg_loss += criterion(output, labels).sum()
          pred = output.data.max(1)[1]
          total_correct += pred.eq(labels.data.view_as(pred)).sum()

  avg_loss /= len(data_test)
  acc = float(total_correct) / len(data_test)
  logprint('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
  return acc

def train_and_test(epoch):
  train(epoch)
  acc = test()
  return acc
 
def main():
  acc = 0; acc_best = 0
  if args.dataset == 'MNIST':
    epoch = 10
  else:
    epoch = 200
  for e in range(1, epoch):
    acc = train_and_test(e)
    if acc > acc_best:
      acc_best = acc
      if args.which_net == "embed":
        torch.save(net,args.output_dir + '/teacher_embed')
      else:
        torch.save(net,args.output_dir + '/teacher')

if __name__ == '__main__':
  main()