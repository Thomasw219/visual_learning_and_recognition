# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    # TODO Q1.2: Adjust data_dir according to where **you** stored the data
    def __init__(self, split, size, data_dir='data/VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        if split == "train":
            self.tf_composition = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(self.size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(3),
            ])
        else:
            self.tf_composition = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(self.size, scale=(1.0, 1.0)),
                transforms.GaussianBlur(3, 2.0),
            ])

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of labels. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        label_list = []
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            # TODO Q1.2: insert your code here, preload labels
            class_array = np.zeros((20,))
            weight_arrays = []
            root = tree.getroot()
            for i, child in enumerate(root):
                if not child.tag == "object":
                    continue
                class_index = self.get_class_index(child[0].text)
                class_array[class_index] = 1
                if int(child[3].text) != 1:
                    weight_array = np.ones((20,))
                    weight_array[class_index] = 0
                    weight_arrays.append(weight_array)
            label_list.append([class_array, np.logical_or.reduce(weight_arrays)])

        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]
        fpath = os.path.join(self.img_dir, findex + '.jpg')
        # TODO Q1.2: insert your code here. hint: read image, find the labels and weight.
        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]
        img = Image.open(fpath)

        image = self.tf_composition(img)
        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)
        return image, label, wgt
