# -*- coding:utf-8 -*-
"""
@file name  : brain_mri_dataset.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-03-03
@brief      : brain mri 数据集读取
"""
import os
import random

import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from skimage.io import imread


# 读取中文路径的图片
def cv_imread(path_file):
    """
    使用OpenCV读取图片，支持中文路径
    
    问题背景：
    - cv2.imread() 无法直接读取包含中文字符的路径
    - 使用np.fromfile() + cv2.imdecode() 的组合可以解决这个问题
    
    @param path_file: 图片文件的路径，支持中文路径
    @return: 读取的图片数据，格式为numpy数组
    """
    # 使用numpy读取文件为字节流，然后通过cv2.imdecode解码为图片
    # np.fromfile(path_file, dtype=np.uint8): 将文件读取为uint8类型的字节数组
    # cv2.IMREAD_UNCHANGED: 按原始格式读取图片，包括alpha通道
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


class BrainMRIDataset(Dataset):
    """
    脑部MRI数据集类：用于加载和处理脑部MRI图像及其对应的分割掩码
    
    功能特点：
    1. 支持CSV格式的数据索引文件
    2. 自动处理图像和掩码的对应关系
    3. 支持数据增强变换
    4. 自动转换掩码标签格式
    """
    
    def __init__(self, path_csv, transforms_=None):
        """
        初始化数据集
        
        @param path_csv: CSV文件的路径，包含图像和掩码的文件路径信息
        @param transforms_: 数据增强变换，可选参数
        """
        # 读取CSV文件，获取数据索引信息
        # CSV文件通常包含：索引列、图像路径列、掩码路径列
        self.df = pd.read_csv(path_csv)
        # 保存数据增强变换器
        self.transforms = transforms_

    def __len__(self):
        """
        返回数据集的长度
        
        @return: 数据集中的样本数量
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        @param idx: 样本索引
        @return: 返回元组 (image, mask)
                - image: 处理后的图像数据
                - mask: 处理后的掩码标签
        """
        # 读取图像：从CSV的第1列（索引为1）获取图像路径
        image = cv_imread(self.df.iloc[idx, 1])
        # 读取掩码：从CSV的第2列（索引为2）获取掩码路径
        mask = cv_imread(self.df.iloc[idx, 2])
        
        # 标签转换：将掩码中的255值转换为1，实现二分类标签
        # 原始掩码：0表示背景，255表示前景
        # 转换后：0表示背景，1表示前景
        mask[mask == 255] = 1  # 转换为0, 1 二分类标签

        # 应用数据增强变换（如果提供了变换器）
        if self.transforms:
            # 使用albumentations库进行数据增强
            # 同时变换图像和掩码，保持它们的一致性
            augmented = self.transforms(image=image, mask=mask)
            # 从增强结果中提取变换后的图像和掩码
            image, augmented["image"]
            mask = augmented["mask"]

        # 返回图像和掩码，将掩码转换为long类型（适用于PyTorch）
        return image, mask.long()


if __name__ == "__main__":
    """
    主程序入口：用于测试数据集类的功能
    
    测试内容：
    1. 创建训练集和验证集实例
    2. 构建数据加载器
    3. 遍历数据并打印形状信息
    """
    # 设置数据路径
    root_dir_train = r"../data_train.csv"  # 训练数据CSV文件路径
    root_dir_valid = r"../data_val.csv"    # 验证数据CSV文件路径

    # 创建数据集实例
    train_set = BrainMRIDataset(root_dir_train)  # 训练集
    valid_set = BrainMRIDataset(root_dir_valid)  # 验证集

    # 创建数据加载器：批次大小为2，随机打乱
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    
    # 遍历训练数据，测试数据加载功能
    for i, (inputs, target) in enumerate(train_loader):
        # 打印批次索引、输入形状、输入数据、目标形状、目标数据
        print(i, inputs.shape, inputs, target.shape, target)
