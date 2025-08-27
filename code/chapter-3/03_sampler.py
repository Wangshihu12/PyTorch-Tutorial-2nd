# -*- coding:utf-8 -*-
"""
@file name  : 03_sampler.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : WeightedRandomSampler使用，初认识
@description: 演示如何使用PyTorch的WeightedRandomSampler来处理不平衡数据集
             通过给不同类别分配不同权重，实现有偏采样，解决类别不平衡问题
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms


class COVID19Dataset(Dataset):
    """
    COVID-19数据集类，继承自PyTorch的Dataset类
    用于加载和处理COVID-19医学图像数据
    """
    
    def __init__(self, root_dir, txt_path, transform=None):
        """
        初始化数据集
        
        @param root_dir: 图像文件所在的根目录路径
        @param txt_path: 包含图像路径和标签信息的文本文件路径
        @param transform: 图像预处理变换操作，默认为None
        """
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_info = []  # 存储图像路径和标签的列表，格式：[(path, label), ...]
        self.label_array = None
        self._get_img_info()  # 调用私有方法读取图像信息

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: 样本索引
        @return: 返回预处理后的图像张量和对应的标签
        """
        path_img, label = self.img_info[index]  # 根据索引获取图像路径和标签
        img = Image.open(path_img).convert('L')  # 打开图像并转换为灰度图

        if self.transform is not None:
            img = self.transform(img)  # 应用图像预处理变换

        return img, label

    def __len__(self):
        """
        返回数据集的样本总数
        
        @return: 数据集中的样本数量
        @raises Exception: 当数据集为空时抛出异常
        """
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        私有方法：读取文本文件，解析图像路径和标签信息
        将硬盘中的数据路径和标签读取到内存中，存储在img_info列表中
        
        @return: 无返回值，直接修改self.img_info
        """
        # 读取txt文件，解析其中的数据
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip()  # 读取文件内容并去除首尾空白字符
            txt_data = txt_data.split("\n")  # 按行分割文本

        # 解析每行数据，提取图像路径和标签
        # 假设txt格式为：图像名 其他信息 标签
        self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                         for i in txt_data]


if __name__ == "__main__":
    # 主程序入口：演示WeightedRandomSampler的使用
    
    # 设置数据集路径
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-demo"  # COVID-19数据集根目录
    img_dir = os.path.join(root_dir, "imgs")  # 图像文件目录
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")  # 训练标签文件路径

    # 设置数据预处理变换
    normalize = transforms.Normalize([0.5], [0.5])  # 标准化变换，均值为0.5，标准差为0.5
    transforms_train = transforms.Compose([
        transforms.Resize((4, 4)),  # 将图像调整为4x4大小
        transforms.ToTensor(),      # 转换为PyTorch张量
        normalize                   # 应用标准化
    ])
    
    # 创建COVID19Dataset实例
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_train)

    # 第一步：计算每个类别的采样权重
    # 类别0的权重为1，类别1的权重为5，表示类别1的采样概率是类别0的5倍
    weights = torch.tensor([1, 5], dtype=torch.float)

    # 第二步：根据每个样本的标签，生成对应的采样权重
    train_targets = [sample[1] for sample in train_data.img_info]  # 提取所有样本的标签
    samples_weights = weights[train_targets]  # 根据标签映射到对应的权重

    # 第三步：实例化WeightedRandomSampler
    # 这个采样器会根据权重进行有偏随机采样，解决类别不平衡问题
    sampler_w = WeightedRandomSampler(
        weights=samples_weights,        # 每个样本的采样权重
        num_samples=len(samples_weights),  # 采样样本总数
        replacement=True)               # 允许重复采样（有放回采样）

    # 设置DataLoader，使用自定义的采样器
    train_loader = DataLoader(dataset=train_data, batch_size=2, sampler=sampler_w)
    
    # 演示采样效果：运行10个epoch，观察采样结果
    for epoch in range(10):
        for i, (inputs, target) in enumerate(train_loader):
            print(target.shape, target)  # 打印每个batch的标签形状和内容
    
    # 说明：由于是有放回采样，并且类别1的采样概率比类别0高5倍
    # 所以在采样结果中会看到很多次出现[1, 1]的情况
    print("\n由于是有放回采样，并且样本1的采样概率比0高5倍，可以看到很多次出现[1, 1]")









