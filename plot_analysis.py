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
import time
TimeID = time.strftime("%Y%m%d-%H%M%S")
from my_utils import check_path
"""
python  thisfile  /path/to/log1.txt  /path/to/log2.txt
"""
logs = sys.argv[1:]
for log in logs:
  loss_oh = []; loss_ie = []; loss_a = []; loss_kd = []; acc = []
  log = check_path(log)
  ie_lw = log.split("ablation_ie_")[1].split("/")[0]
  for line in open(log):
    if "[loss_oh: " in line:
      loss_oh.append(float(line.split("loss_oh:")[1].split("]")[0].strip()))
      loss_oh.append(float(line.split("loss_ie:")[1].split("]")[0].strip()))
      loss_a.append(float(line.split("loss_a:")[1].split("]")[0].strip()))
      loss_kd.append(float(line.split("loss_kd:")[1].split("]")[0].strip()))
    elif "Accuracy: " in line:
      acc.append(float(line.strip().split(" ")[-1]))
  plt.plot(acc, label="ie_weight = "+ie_lw)
plt.legend()
plt.grid()
plt.xlim([-2, 50])
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.savefig("ablate_ie.jpg")