import torch
import torch.nn as nn
Conv2d     = nn.Conv2d
ReLU       = nn.ReLU
MaxPool2d  = nn.MaxPool2d
Dropout    = nn.Dropout
Linear     = nn.Linear
Sequential = nn.Sequential

class AlexNet(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet, self).__init__()
    self.features = Sequential(
      Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.classifier = Sequential(
      Dropout(p=0.5),
      Linear(in_features=9216, out_features=4096, bias=True),
      ReLU(inplace=True),
      Dropout(p=0.5),
      Linear(in_features=4096, out_features=4096, bias=True),
      ReLU(inplace=True),
    )
    num_attributes = 40 # celeba
    for i in range(num_attributes):
      setattr(self, 'end' + str(i).zfill(2), nn.Linear(4096, 2))
    self.num_attributes = num_attributes
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for p in self.parameters():
        p.require_grad = False
  
  def forward(self, x, embed=False):
    if embed:
      x = self.features(x)
      x = x.view(x.size(0), -1)
      embed = self.classifier[:6](x) # upto and include ReLU
      y = []
      for i in range(self.num_attributes):
        y.append(eval("self.end" + str(i).zfill(2))(embed))
      return y, embed
    else:
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      y = []
      for i in range(self.num_attributes):
        subnet = getattr(self, 'end' + str(i).zfill(2))
        y.append(subnet(x))
      return y
      
class AlexNet_half(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet_half, self).__init__()
    self.features = Sequential(
      Conv2d(3, 32, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      ReLU(inplace=True),
      MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.classifier = Sequential(
      Dropout(p=0.5),
      Linear(in_features=4608, out_features=4096, bias=True),
      ReLU(inplace=True),
      Dropout(p=0.5),
      Linear(in_features=4096, out_features=4096, bias=True),
      ReLU(inplace=True),
      Linear(in_features=4096, out_features=40, bias=True),
    )
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for p in self.parameters():
        p.require_grad = False
  
  def forward(self, x, embed=False):
    if embed:
      x = self.features(x)
      embed = self.classifier[:6](x) # upto and include ReLU
      x = self.classifier[6:](embed)
      return x, embed
    else:
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x