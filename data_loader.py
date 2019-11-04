import torch.utils.data as data
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms as transforms
import os
import numpy as np
import torch

def is_img(x):
  _, ext = os.path.splitext(x)
  if ext.lower() in ['.jpg', '.png', '.bmp', '.jpeg']:
    return True
  else:
    return False

class CelebA(data.Dataset):
  def __init__(self, img_dir, label_file, transform):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if is_img(i)]
    self.transform = transform
    self.label = {}
    num_attributes = 40
    for line in open(label_file):
      if ".jpg" not in line: continue
      img_name, *attr = line.strip().split()
      label = torch.zeros(num_attributes).long()
      for i in range(num_attributes):
        if attr[i] == "1":
          label[i] = 1
      self.label[img_name] = label
  def __getitem__(self, index):
    img_path = self.img_list[index]
    img_name = img_path.split("/")[-1]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224)) # for alexnet
    img = self.transform(img)
    return img.squeeze(0), self.label[img_name]

  def __len__(self):
    return len(self.img_list)