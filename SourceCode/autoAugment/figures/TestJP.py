from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import math
import random
import pdb
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision

img = Image.open("E:\\AutoAugment-master\\AutoAugment-master\\_001a_0.jpg")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.imshow(img)
img2 = Image.open("E:\\AutoAugment-master\\AutoAugment-master\\_001a_0.jpg")
ax2.imshow(img2)
plt.show()

def show_sixteen(images, titles=0):
    f, axarr = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx])
        ax.axis("off")
        if titles: ax.set_title(titles[idx])
    plt.show()

policy = ImageNetPolicy()

imgs = []
for _ in range(8): imgs.append(policy(img))
for _ in range(8): imgs.append(policy(img2))
show_sixteen(imgs)

data = ImageFolder("C:\\Users\\hjnui\\Desktop\\BankSet")
loader = DataLoader(data, batch_size=1)

imgs, count = [], 0
count = 0

for _ in range(10):
    for img in loader:
        img = np.transpose(img[0][0].numpy()*255, (1,2,0)).astype(np.uint8)
        imgs.append(img)
        count += 1
        if count == 16:
            show_sixteen(imgs)
            imgs, count = [], 0

