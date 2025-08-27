# -*- coding:utf-8 -*-
"""
@file name  : 04_dataset_api.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : dataset api 使用： concat\sub_set\random_split\
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from PIL import Image
from torchvision.transforms import transforms


class COVID19Dataset(Dataset):
    """
    COVID-19数据集类，继承自PyTorch的Dataset类
    用于处理COVID-19医学图像数据，支持从txt文件读取图像路径和标签
    """
    
    def __init__(self, root_dir, txt_path, transform=None):
        """
        初始化COVID-19数据集
        
        @param root_dir: str, 图像文件的根目录路径
        @param txt_path: str, 包含图像路径和标签信息的txt文件路径
        @param transform: 图像预处理变换，默认为None
        """
        self.root_dir = root_dir          # 存储图像根目录
        self.txt_path = txt_path          # 存储标签文件路径
        self.transform = transform        # 存储图像预处理变换
        self.img_info = []                # 存储图像信息列表，格式：[(path, label), ...]
        self.label_array = None           # 标签数组（当前未使用）
        self._get_img_info()              # 调用私有方法读取图像信息

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: int, 数据样本的索引
        @return: tuple, 返回(图像张量, 标签)的元组
        """
        # 根据索引获取图像路径和标签
        path_img, label = self.img_info[index]
        # 打开图像并转换为灰度图（L模式）
        img = Image.open(path_img).convert('L')

        # 如果定义了预处理变换，则应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        获取数据集的总样本数量
        
        @return: int, 数据集中样本的总数
        """
        # 检查数据集是否为空，如果为空则抛出异常
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        私有方法：从txt文件中读取图像路径和标签信息
        将硬盘中的数据路径和标签读取到img_info列表中
        
        @return: None, 直接修改self.img_info属性
        """
        # 读取txt文件内容
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip()           # 读取文件内容并去除首尾空白
            txt_data = txt_data.split("\n")      # 按行分割

        # 解析每行数据，提取图像路径和标签
        # 假设txt格式为：图像名 其他信息 标签
        # 将图像路径和标签组成元组存储在img_info列表中
        self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                         for i in txt_data]


class COVID19Dataset_3(Dataset):
    """
    COVID-19数据集类（形式3），继承自PyTorch的Dataset类
    用于处理COVID-19医学图像数据，支持从CSV文件读取图像路径和标签
    数据集的训练/验证划分信息存储在CSV文件中
    """

    def __init__(self, root_dir, path_csv, mode, transform=None):
        """
        初始化COVID-19数据集（形式3）
        
        @param root_dir: str, 图像文件的根目录路径
        @param path_csv: str, 包含数据集元信息的CSV文件路径
        @param mode: str, 数据集模式，可选值：'train' 或 'valid'
        @param transform: 图像预处理变换，默认为None
        """
        self.root_dir = root_dir          # 存储图像根目录
        self.path_csv = path_csv          # 存储CSV文件路径
        self.mode = mode                  # 存储数据集模式（训练/验证）
        self.transform = transform        # 存储图像预处理变换
        self.img_info = []                # 存储图像信息列表，格式：[(path, label), ...]
        self.label_array = None           # 标签数组（当前未使用）
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self._get_img_info()              # 调用私有方法读取图像信息

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: int, 数据样本的索引
        @return: tuple, 返回(图像张量, 标签)的元组
        """
        # 根据索引获取图像路径和标签
        path_img, label = self.img_info[index]
        # 打开图像并转换为灰度图（L模式）
        img = Image.open(path_img).convert('L')

        # 如果定义了预处理变换，则应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        获取数据集的总样本数量
        
        @return: int, 数据集中样本的总数
        """
        # 检查数据集是否为空，如果为空则抛出异常
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        私有方法：从CSV文件中读取图像路径和标签信息
        根据指定的模式（训练/验证）过滤数据，将硬盘中的数据路径和标签读取到img_info列表中
        
        @return: None, 直接修改self.img_info属性
        """
        # 读取CSV文件
        df = pd.read_csv(self.path_csv)
        # 根据set-type列过滤出指定模式的数据（训练或验证）
        df.drop(df[df["set-type"] != self.mode].index, inplace=True)  # 只要对应的数据
        df.reset_index(inplace=True)    # 非常重要！ pandas的drop不会改变index
        
        # 遍历表格，获取每张样本信息
        for idx in range(len(df)):
            # 构建完整的图像路径
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            # 获取标签并转换为整数类型
            label_int = int(df.loc[idx, "label"])
            # 将图像路径和标签添加到img_info列表中
            self.img_info.append((path_img, label_int))


class COVID19Dataset_2(Dataset):
    """
    COVID-19数据集类（形式2），继承自PyTorch的Dataset类
    用于处理COVID-19医学图像数据，支持从文件夹结构自动推断标签信息
    数据集的划分和标签通过文件夹名称体现
    """

    def __init__(self, root_dir, transform=None):
        """
        初始化COVID-19数据集（形式2）
        
        @param root_dir: str, 图像文件的根目录路径，子文件夹名称表示标签
        @param transform: 图像预处理变换，默认为None
        """
        self.root_dir = root_dir          # 存储图像根目录
        self.transform = transform        # 存储图像预处理变换
        self.img_info = []                # 存储图像信息列表，格式：[(path, label), ...]
        self.label_array = None           # 标签数组（当前未使用）
        
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        # 文件夹名称到标签值的映射
        self.str_2_int = {"no-finding": 0, "covid-19": 1}

        self._get_img_info()              # 调用私有方法读取图像信息

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: int, 数据样本的索引
        @return: tuple, 返回(图像张量, 标签)的元组
        """
        # 根据索引获取图像路径和标签
        path_img, label = self.img_info[index]
        # 打开图像并转换为灰度图（L模式）
        img = Image.open(path_img).convert('L')

        # 如果定义了预处理变换，则应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        获取数据集的总样本数量
        
        @return: int, 数据集中样本的总数
        """
        # 检查数据集是否为空，如果为空则抛出异常
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        私有方法：通过遍历文件夹结构自动获取图像路径和标签信息
        将硬盘中的数据路径和标签读取到img_info列表中
        
        @return: None, 直接修改self.img_info属性
        """
        # 遍历根目录下的所有子目录和文件
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # 只处理PNG和JPEG格式的图像文件
                if file.endswith("png") or file.endswith("jpeg"):
                    # 构建完整的图像文件路径
                    path_img = os.path.join(root, file)
                    # 获取子目录名称（即标签名称）
                    sub_dir = os.path.basename(root)
                    # 将字符串标签转换为整数标签
                    label_int = self.str_2_int[sub_dir]
                    # 将图像路径和标签添加到img_info列表中
                    self.img_info.append((path_img, label_int))


if __name__ == "__main__":
    # =============================== 数据集连接（Concat）示例 ================================
    # 创建第一个数据集：从txt文件读取数据
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")                           # 图像目录路径
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")     # 训练标签文件路径
    train_data_1 = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train)  # 创建数据集实例
    
    # 创建第二个数据集：从文件夹结构读取数据
    root_dir_train = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-2\train"  # path to your data
    train_data_2 = COVID19Dataset_2(root_dir_train)                    # 创建数据集实例
    
    # 创建第三个数据集：从CSV文件读取数据
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\imgs"  # path to your data
    path_csv = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\dataset-meta-data.csv"
    train_data_3 = COVID19Dataset_3(root_dir, path_csv, "train")       # 创建数据集实例
    
    # 使用ConcatDataset连接三个数据集
    train_set_all = ConcatDataset([train_data_1, train_data_2, train_data_3])
    print(len(train_set_all))  # 输出：2 + 2 + 2 = 6（三个数据集的样本总数）

    # =============================== 子集（Subset）示例 ================================
    # 从连接后的数据集中提取指定索引的样本，创建子数据集
    train_sub_set = Subset(train_set_all, [0, 1, 2, 5])  # 将这4个样本抽出来构成子数据集
    print(len(train_sub_set))                              # 输出：4（子数据集的样本数量）
    
    # =============================== 随机分割（Random Split）示例 ================================
    # 将连接后的数据集随机分割为两个子集
    set_split_1, set_split_2 = random_split(train_set_all, [4, 2])  # 分割为4个和2个样本的两个子集
    print(len(set_split_1), len(set_split_2))                        # 输出：4 2（两个分割子集的样本数量）
