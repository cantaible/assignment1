# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:58:12 2018

@author: 11854
"""

from __future__ import print_function

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt



cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass
#导入数据集
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#类的数量
num_classes=len(classes)
#每个类有多少样本
samples_per_class=7
for y,cls in enumerate(classes):
    #该函数输入一个矩阵，返回扁平化后矩阵中非零元素的位置（index）
    idxs=np.flatnonzero(y_train==y)
    #random.choice方法返回一个列表，元组或字符串的随机项
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        #一共有samples_per_class个图像，num_classes行，现在在画第plt_idx个
        plt.subplot(samples_per_class, num_classes, plt_idx)
        #
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()