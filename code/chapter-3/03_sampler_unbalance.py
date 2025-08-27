# -*- coding:utf-8 -*-
"""
@file name  : 03_sampler_unbalance.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : WeightedRandomSampler  使用于不均衡数据集
"""
import os
import shutil
import collections
import torch
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms


def make_fake_data(base_num):
    """
    制作不平衡的虚拟数据集，用于演示WeightedRandomSampler的使用
    
    该函数从CIFAR-10数据集中随机选择不同数量的样本，创建类别不平衡的数据集
    每个类别的样本数量为：(类别索引 + 1) * base_num
    
    @param base_num: int, 基础样本数量，用于计算每个类别的样本数量
    @return: None, 直接创建不平衡数据集文件
    """
    # 源数据目录：CIFAR-10训练集路径
    root_dir = r"G:\deep_learning_data\cifar10\cifar10_train"
    # 输出目录：不平衡数据集的保存路径
    out_dir = r"E:\pytorch-tutorial-2nd\data\datasets\cifar-unbalance"

    # 导入random模块（注意：这里应该放在文件顶部）
    import random
    
    # 遍历10个类别（CIFAR-10有10个类别）
    for i in range(10):
        # 计算当前类别的样本数量：(类别索引 + 1) * base_num
        # 这样设计会导致类别0样本最少，类别9样本最多，形成不平衡
        sample_num = (i + 1) * base_num

        # 构建源数据子目录路径（类别i的文件夹）
        sub_dir = os.path.join(root_dir, str(i))
        # 获取该类别下的所有图像文件名
        path_imgs = os.listdir(sub_dir)
        # 随机打乱图像文件顺序，确保随机选择
        random.shuffle(path_imgs)

        # 构建输出子目录路径
        out_sub_dir = os.path.join(out_dir, str(i))
        # 如果输出目录不存在，则创建
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)

        # 从打乱后的文件列表中复制指定数量的样本
        for j in range(sample_num):
            # 获取当前要复制的文件名
            file_name = path_imgs[j]
            # 构建源图像的完整路径
            path_img = os.path.join(sub_dir, file_name)
            # 复制图像文件到输出目录
            shutil.copy(path_img, out_sub_dir)
    
    # 打印完成提示
    print("done")


class CifarDataset(Dataset):
    """
    CIFAR数据集类，继承自PyTorch的Dataset类
    用于加载和处理CIFAR格式的图像数据，支持从文件夹结构自动推断标签
    """
    
    # 类别名称列表，对应CIFAR-10的10个类别
    names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 类别总数
    cls_num = len(names)

    def __init__(self, root_dir, transform=None):
        """
        初始化CIFAR数据集
        
        @param root_dir: str, 数据集的根目录路径，子文件夹名称表示类别标签
        @param transform: 图像预处理变换，默认为None
        """
        self.root_dir = root_dir          # 存储数据集根目录
        self.transform = transform        # 存储图像预处理变换
        self.img_info = []                # 定义list用于存储样本路径、标签
        self._get_img_info()              # 调用私有方法读取图像信息

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: int, 数据样本的索引
        @return: tuple, 返回(图像张量, 标签)的元组
        """
        # 根据索引获取图像路径和标签
        path_img, label = self.img_info[index]
        # 打开图像并转换为RGB格式
        img = Image.open(path_img).convert('RGB')

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
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        私有方法：从文件夹结构中读取图像路径和标签信息
        遍历根目录下的所有子目录，自动推断类别标签
        
        @return: None, 直接修改self.img_info属性
        """
        # 遍历根目录下的所有子目录和文件
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历每个类别子目录
            for sub_dir in dirs:
                # 获取该类别下的所有图像文件名
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 过滤出PNG格式的图像文件
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                
                # 遍历该类别下的所有图像
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 构建图像的绝对路径
                    path_img = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    # 将子目录名称转换为整数标签
                    label = int(sub_dir)
                    # 将图像路径和标签添加到img_info列表中
                    self.img_info.append((path_img, int(label)))
        
        # 将数据顺序打乱，提高训练效果
        random.shuffle(self.img_info)


if __name__ == "__main__":
    # 注释掉数据生成函数调用
    # make_fake_data()
    
    # 数据集下载链接：https://pan.baidu.com/s/1ST85f8qgyKQucvKCBKbzug
    # 提取码：vf4j
    
    # 设置不平衡数据集的根目录路径
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\cifar-unbalance"

    # 定义图像预处理变换
    # 使用CIFAR-10数据集的标准化参数
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),     # 调整图像尺寸为32x32
        transforms.ToTensor(),           # 转换为PyTorch张量
        normalize                        # 应用标准化
    ])
    
    # 创建CIFAR数据集实例
    train_data = CifarDataset(root_dir=root_dir, transform=transforms_train)

    # =============================== 第一步：计算各类别的采样权重 ================================
    # 提取所有样本的标签
    train_targets = [sample[1] for sample in train_data.img_info]
    # 统计每个类别的样本数量
    label_counter = collections.Counter(train_targets)
    # 按类别索引排序，获取每个类别的样本数量列表
    # 需要特别注意，此list的顺序必须与类别索引对应！
    class_sample_counts = [label_counter[k] for k in sorted(label_counter)]
    
    # 计算权重，利用倒数即可
    # 样本数量越少的类别，权重越大，这样可以在采样时增加稀有类别的出现概率
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 也可以使用其他权重计算方式，比如乘以一个常数
    # weights = 12345. / torch.tensor(class_sample_counts, dtype=torch.float)

    # =============================== 第二步：生成每个样本的采样权重 ================================
    # 为每个样本分配对应的类别权重
    # train_targets中的每个标签对应weights中的权重值
    samples_weights = weights[train_targets]

    # =============================== 第三步：实例化WeightedRandomSampler ================================
    # 创建加权随机采样器
    sampler_w = WeightedRandomSampler(
        weights=samples_weights,         # 每个样本的权重
        num_samples=len(samples_weights), # 采样样本总数
        replacement=True)                # 允许重复采样

    # =============================== 配置DataLoader ================================
    # 使用加权采样器的DataLoader
    train_loader_sampler = DataLoader(dataset=train_data, batch_size=16, sampler=sampler_w)
    # 使用默认随机采样的DataLoader（作为对比）
    train_loader = DataLoader(dataset=train_data, batch_size=16)

    def show_sample(loader):
        """
        显示数据加载器中各类别的样本分布情况
        
        @param loader: DataLoader, 数据加载器
        @return: None, 直接打印统计结果
        """
        # 运行10个epoch来观察样本分布
        for epoch in range(10):
            label_count = []  # 存储当前epoch中所有样本的标签
            # 遍历数据加载器中的所有批次
            for i, (inputs, target) in enumerate(loader):
                # 将当前批次的标签添加到列表中
                label_count.extend(target.tolist())
            # 统计并打印各类别的样本数量
            print(collections.Counter(label_count))

    # 显示使用默认采样器的数据分布（不平衡）
    print("使用默认采样器的数据分布（不平衡）：")
    show_sample(train_loader)
    
    # 显示使用加权采样器的数据分布（平衡）
    print("\n接下来运用sampler（加权采样器）\n")
    show_sample(train_loader_sampler)





