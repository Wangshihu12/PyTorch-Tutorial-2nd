# -*- coding:utf-8 -*-
"""
@file name  : 02_dataloader.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-22
@brief      : dataloader使用学习
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class AntsBeesDataset(Dataset):
    """
    蚂蚁蜜蜂分类数据集类
    继承自PyTorch的Dataset，用于加载hymenoptera（膜翅目昆虫）数据集
    数据集组织方式：按类别分文件夹存储
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录，包含按类别组织的子文件夹
            transform: 数据预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # 存储图像路径和标签的列表 [(path, label), ...]
        self.label_array = None
        
        # 创建字符串标签到整数标签的映射字典
        # 将文件夹名称转换为模型训练时使用的整数标签
        # ants -> 0, bees -> 1
        self.str_2_int = {"ants": 0, "bees": 1}
        
        # 调用方法读取图像信息
        self._get_img_info()

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        Args:
            index: 样本索引
        Returns:
            img: 预处理后的图像张量
            label: 对应的整数标签
        """
        path_img, label = self.img_info[index]  # 获取图像路径和标签
        img = Image.open(path_img)  # 打开图像（注意：这里没有转换为灰度图，保持RGB）

        if self.transform is not None:
            img = self.transform(img)  # 应用数据预处理变换

        return img, label

    def __len__(self):
        """
        返回数据集的样本数量
        """
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，通过遍历文件夹结构自动获取图像路径和对应的标签
        工作流程：
        1. 遍历根目录下的所有子文件夹
        2. 子文件夹名称对应类别标签（ants或bees）
        3. 将图像路径和标签信息存储在self.img_info中
        """
        # 使用os.walk遍历根目录下的所有子目录和文件
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # 只处理jpg格式的图像文件
                if file.endswith("jpg"):
                    # 构建完整的图像文件路径
                    path_img = os.path.join(root, file)
                    
                    # 获取图像所在的子文件夹名称，即类别标签
                    sub_dir = os.path.basename(root)
                    
                    # 将字符串标签转换为整数标签
                    label_int = self.str_2_int[sub_dir]
                    
                    # 将图像路径和标签添加到列表中
                    self.img_info.append((path_img, label_int))


if __name__ == "__main__":
    # 数据集下载链接
    # 链接：https://pan.baidu.com/s/1X11v5XEbdrgdgsVAESCVrA
    # 提取码：4wx1
    
    # 设置数据集路径
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\mini-hymenoptera_data\train"
    
    # =========================== 配合 DataLoader 使用 ===================================
    # 定义数据预处理变换
    # 标准化参数来自ImageNet数据集的统计值，这是预训练模型常用的标准化参数
    # 均值：[0.485, 0.456, 0.406] - RGB三个通道的均值
    # 标准差：[0.229, 0.224, 0.225] - RGB三个通道的标准差
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # 组合多个预处理步骤
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),    # 将图像调整为224x224像素（ImageNet标准尺寸）
        transforms.ToTensor(),            # 转换为PyTorch张量，范围[0,1]
        normalize                         # 标准化，使用ImageNet统计值
    ])

    # 创建带有预处理变换的训练集
    train_set = AntsBeesDataset(root_dir, transform=transforms_train)

    # 创建不同配置的DataLoader，展示不同参数的效果
    # DataLoader 1: 批次大小为2，随机打乱
    train_loader_bs2 = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    
    # DataLoader 2: 批次大小为3，随机打乱
    train_loader_bs3 = DataLoader(dataset=train_set, batch_size=3, shuffle=True)
    
    # DataLoader 3: 批次大小为2，随机打乱，丢弃最后一个不完整的批次
    # drop_last=True: 当数据集大小不能被batch_size整除时，丢弃最后一个不完整的批次
    train_loader_bs2_drop = DataLoader(dataset=train_set, batch_size=2, shuffle=True, drop_last=True)

    # 遍历不同的DataLoader，观察批处理效果
    print("=== DataLoader with batch_size=2 ===")
    for i, (inputs, target) in enumerate(train_loader_bs2):
        # i: 批次索引
        # inputs: 输入图像张量，形状为[batch_size, channels, height, width]
        # target: 目标标签张量，形状为[batch_size]
        print(i, inputs.shape, target.shape, target)
    
    print("\n=== DataLoader with batch_size=3 ===")
    for i, (inputs, target) in enumerate(train_loader_bs3):
        print(i, inputs.shape, target.shape, target)
    
    print("\n=== DataLoader with batch_size=2, drop_last=True ===")
    for i, (inputs, target) in enumerate(train_loader_bs2_drop):
        print(i, inputs.shape, target.shape, target)

