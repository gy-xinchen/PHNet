# -*- coding: utf-8 -*-
# @Time    : 2023/2/7 9:50
# @Author  : yuan
# @File    : CreateniiDataset.py
# -*- coding: utf-8 -*-
# @Time    : 2022/12/16 13:30
# @Author  : yuan
# @File    : CreateNiiDataset.py

import os

import SimpleITK as sitk
import numpy as np
import torch
import torchio as tio

class CreateNiiDataset():

    def __init__(self, row_train, row_val, train=True, transform=None, val_transform=None ):
        self.train = train
        if self.train:
            self.train_str = row_train
            self.transform = transform
        else:
            self.val_str = row_val
            self.val_transform = val_transform


    # iter
    def __getitem__(self, item):
        if self.train:
            train_str = self.train_str[item]
            train_value = (eval(train_str))['image'] # tuple:2
            img_train = train_value[0]
            train_label = train_value[1]
            # anchor_id = train_value[2]
            img_train = sitk.ReadImage(img_train)
            data_train = sitk.GetArrayFromImage(img_train)
            if self.transform is not None:
                data_train = self.transform(data_train)
            data_train = data_train[np.newaxis, np.newaxis, :, :]
            data_train = data_train/1.0 # 用于将uint18转换为float
            data_train_tensor = torch.from_numpy(data_train)
            data_train_tensor = data_train_tensor.type(torch.FloatTensor)
            data_train_tensor =data_train_tensor.squeeze(0)

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=data_train_tensor),
                label=train_label,
                # anchor=anchor_id
            )
            return subject
        else:
            val_str = self.val_str[item]
            val_value = (eval(val_str))['image']  # tuple:2
            img_val = val_value[0]
            val_label = val_value[1]
            img_val   = sitk.ReadImage(img_val)
            data_val   = sitk.GetArrayFromImage(img_val)
            if self.val_transform is not None:
                data_val   = self.val_transform(data_val)
            data_val = data_val[np.newaxis, np.newaxis, :, :]
            data_val = data_val/1.0 # 用于将uint18转换为float
            data_val_tensor = torch.from_numpy(data_val)
            data_val_tensor = data_val_tensor.type(torch.FloatTensor)
            data_val_tensor = data_val_tensor.squeeze(0)

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=data_val_tensor),
                label=val_label
            )
            return subject

    def load_data(self):
        return self

    # return dataset
    def __len__(self):
        if self.train:
            return len(self.train_str)
        else:
            return len(self.val_str)




