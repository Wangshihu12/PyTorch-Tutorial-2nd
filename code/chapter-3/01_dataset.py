# -*- coding:utf-8 -*-
"""
@file name  : 01_datasets.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-18
@brief      : dataset编写示例
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class COVID19Dataset_2(Dataset):
    """
    对应数据集形式-2：数据的划分及标签在文件夹中体现
    这种组织方式将不同类别的图像放在不同的文件夹中，文件夹名即为类别标签
    例如：
    root_dir/
    ├── no-finding/     # 正常图像文件夹
    │   ├── img1.png
    │   └── img2.png
    └── covid-19/       # COVID-19图像文件夹
        ├── img3.png
        └── img4.png
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
        # 这种映射方式使得模型可以处理分类问题
        self.str_2_int = {"no-finding": 0, "covid-19": 1}

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
        img = Image.open(path_img).convert('L')  # 打开图像并转换为灰度图

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
        实现数据集的读取，将硬盘中的数据路径和标签读取进来，存储在列表中
        通过遍历文件夹结构自动获取图像路径和对应的标签
        
        工作流程：
        1. 遍历根目录下的所有子文件夹
        2. 子文件夹名称对应类别标签
        3. 将图像路径和标签信息存储在self.img_info中
        
        Returns:
            None，但会填充self.img_info列表
        """
        # 使用os.walk遍历根目录下的所有子目录和文件
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # 只处理图像文件（png或jpeg格式）
                if file.endswith("png") or file.endswith("jpeg"):
                    # 构建完整的图像文件路径
                    path_img = os.path.join(root, file)
                    
                    # 获取图像所在的子文件夹名称，即类别标签
                    sub_dir = os.path.basename(root)
                    
                    # 将字符串标签转换为整数标签
                    label_int = self.str_2_int[sub_dir]
                    
                    # 将图像路径和标签添加到列表中
                    self.img_info.append((path_img, label_int))


class COVID19Dataset_3(Dataset):
    """
    对应数据集形式-3：数据的划分及标签在CSV文件中
    这种组织方式将所有数据信息（包括路径、标签、数据集划分）存储在CSV文件中
    通过mode参数来区分训练集和验证集
    """

    def __init__(self, root_dir, path_csv, mode, transform=None):
        """
        初始化数据集
        Args:
            root_dir: 图像文件所在的根目录
            path_csv: CSV文件路径，包含图像名称、标签和数据集划分信息
            mode: 数据集模式，'train' 或 'valid'，用于筛选对应的数据
            transform: 数据预处理变换
        """
        self.root_dir = root_dir
        self.path_csv = path_csv
        self.mode = mode
        self.transform = transform
        self.img_info = []  # 存储图像路径和标签的列表 [(path, label), ...]
        self.label_array = None
        
        # 调用方法读取CSV文件中的图像信息
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
        img = Image.open(path_img).convert('L')  # 打开图像并转换为灰度图

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
        实现数据集的读取，从CSV文件中读取数据路径和标签信息
        通过mode参数筛选对应的训练集或验证集数据
        
        工作流程：
        1. 读取CSV文件
        2. 根据mode筛选对应的数据集划分
        3. 重置索引（重要：pandas的drop操作不会改变索引）
        4. 遍历筛选后的数据，构建图像路径和标签信息
        
        CSV文件格式示例：
        | img-name | label | set-type |
        |----------|-------|----------|
        | img1.png | 0     | train    |
        | img2.png | 1     | train    |
        | img3.png | 0     | valid    |
        | img4.png | 1     | valid    |
        
        Returns:
            None，但会填充self.img_info列表
        """
        # 读取CSV文件到pandas DataFrame
        df = pd.read_csv(self.path_csv)
        
        # 根据mode筛选对应的数据集划分
        # 只保留set-type列等于指定mode的行
        df.drop(df[df["set-type"] != self.mode].index, inplace=True)
        
        # 重置索引：这一步非常重要！
        # pandas的drop操作不会改变原有的索引，可能导致索引不连续
        # reset_index()确保索引从0开始连续排列
        df.reset_index(inplace=True)
        
        # 遍历筛选后的DataFrame，获取每张样本的信息
        for idx in range(len(df)):
            # 构建完整的图像文件路径：root_dir + 图像名称
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            
            # 获取标签并转换为整数类型
            label_int = int(df.loc[idx, "label"])
            
            # 将图像路径和标签添加到列表中
            self.img_info.append((path_img, label_int))


if __name__ == "__main__":

    # =========================== COVID19Dataset_2 使用示例 ===================================
    # 数据集下载链接
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    
    # 设置数据集路径：基于文件夹结构的数据集
    # 训练集和验证集分别在不同的文件夹中
    root_dir_train = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-dataset2&3/covid-19-dataset-2/train"  # 训练集路径
    root_dir_valid = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-dataset2&3/covid-19-dataset-2/valid"  # 验证集路径
    
    # 创建训练集和验证集实例
    # COVID19Dataset_2会自动从文件夹名称中提取标签信息
    train_set = COVID19Dataset_2(root_dir_train)
    valid_set = COVID19Dataset_2(root_dir_valid)

    # =========================== COVID19Dataset_3 使用示例 ===================================
    # 设置数据集路径：基于CSV文件的数据集
    # 所有数据信息都存储在CSV文件中，通过mode参数区分训练集和验证集
    root_dir = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-dataset2&3/covid-19-dataset-3/imgs"  # 图像文件目录
    path_csv = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-dataset2&3/covid-19-dataset-3/dataset-meta-data.csv"  # CSV元数据文件路径
    
    # 创建训练集实例，指定mode为"train"
    train_set = COVID19Dataset_3(root_dir, path_csv, "train")
    
    # 打印数据集信息：数据集大小和第一个样本
    print(len(train_set), next(iter(train_set)))
    print(len(valid_set), next(iter(valid_set)))  # 思考，为什么返回的是 PIL.Image.Image ？
    # 答案：因为我们在__getitem__方法中直接返回了PIL图像，没有应用transform

    # =========================== 配合 DataLoader 使用 ===================================
    # 定义数据预处理变换
    # 标准化：将数据范围从[0,1]调整到[-1,1]，有助于模型训练稳定性
    normalize = transforms.Normalize([0.5], [0.5])  # 均值为0.5，标准差为0.5
    
    # 组合多个预处理步骤
    transforms_train = transforms.Compose([
        transforms.Resize((4, 4)),    # 将图像调整为4x4像素
        transforms.ToTensor(),        # 转换为PyTorch张量，范围[0,1]
        normalize                     # 标准化，范围变为[-1,1]
    ])

    # 创建带有预处理变换的训练集
    # 现在__getitem__方法会返回预处理后的张量，而不是PIL图像
    train_set = COVID19Dataset_3(root_dir, path_csv, "train", transform=transforms_train)
    
    # 创建DataLoader，用于批量加载数据
    # batch_size=2：每次加载2个样本
    # shuffle=True：随机打乱数据顺序，有助于训练稳定性
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    
    # 遍历DataLoader，查看批处理的效果
    for i, (inputs, target) in enumerate(train_loader):
        # i: 批次索引
        # inputs: 输入图像张量，形状为[batch_size, channels, height, width]
        # target: 目标标签张量，形状为[batch_size]
        print(i, inputs.shape, inputs, target.shape, target)
