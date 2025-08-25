# -*- coding:utf-8 -*-
"""
@file name  : 02_COVID_19_cls.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2021-12-28
@brief      : 新冠肺炎X光分类 demo，极简代码实现深度学习模型训练，为后续核心模块讲解，章节内容讲解奠定框架性基础。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def main():
    # 思考：如何实现你的模型训练？第一步干什么？第二步干什么？...第n步...
    # step 1/4 : 数据模块：构建dataset, dataloader，实现对硬盘中数据的读取及设定预处理方法
    # step 2/4 : 模型模块：构建神经网络，用于后续训练
    # step 3/4 : 优化模块：设定损失函数与优化器，用于在训练过程中对网络参数进行更新
    # step 4/4 : 迭代模块: 循环迭代地进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到我们让它停止训练

    # step 1/4 : 数据模块
    class COVID19Dataset(Dataset):
        """
        自定义数据集类，继承自PyTorch的Dataset
        用于加载COVID-19图像数据和对应的标签
        """
        def __init__(self, root_dir, txt_path, transform=None):
            """
            初始化数据集
            Args:
                root_dir: 图像文件所在的根目录
                txt_path: 标签文件路径，包含图像路径和标签信息
                transform: 数据预处理变换
            """
            self.root_dir = root_dir
            self.txt_path = txt_path
            self.transform = transform
            self.img_info = []  # 存储图像路径和标签的列表 [(path, label), ...]
            self.label_array = None
            self._get_img_info()  # 调用方法读取图像信息

        def __getitem__(self, index):
            """
            根据索引获取单个样本
            Args:
                index: 样本索引
            Returns:
                img: 预处理后的图像张量
                label: 对应的标签
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
            读取标签文件，解析图像路径和标签信息
            将信息存储在self.img_info列表中
            """
            # 读取txt文件，解析其中的数据
            with open(self.txt_path, "r") as f:
                txt_data = f.read().strip()  # 读取文件内容并去除首尾空白
                txt_data = txt_data.split("\n")  # 按行分割

            # 解析每一行：格式为 "图像名 标签"
            # 构建完整路径：root_dir + 图像名
            self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                             for i in txt_data]
    
    # 数据集下载链接
    # you can download the datasets from
    # https://pan.baidu.com/s/18BsxploWR3pbybFtNsw5fA  code：pyto
    
    # 设置数据集路径
    root_dir = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")  # 图像文件目录
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")  # 训练集标签文件
    path_txt_valid = os.path.join(root_dir, "labels", "valid.txt")  # 验证集标签文件
    
    # 定义数据预处理变换
    transforms_func = transforms.Compose([
        transforms.Resize((8, 8)),  # 将图像调整为8x8像素
        transforms.ToTensor(),      # 转换为PyTorch张量
    ])
    
    # 创建训练集和验证集数据集对象
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    
    # 创建数据加载器，用于批量加载数据
    train_loader = DataLoader(dataset=train_data, batch_size=2)  # 训练集数据加载器
    valid_loader = DataLoader(dataset=valid_data, batch_size=2)  # 验证集数据加载器

    # step 2/4 : 模型模块
    class TinnyCNN(nn.Module):
        """
        定义一个简单的CNN网络用于COVID-19分类
        网络结构：1个卷积层 + 1个全连接层
        """
        def __init__(self, cls_num=2):
            """
            初始化网络结构
            Args:
                cls_num: 分类类别数，COVID-19为二分类问题
            """
            super(TinnyCNN, self).__init__()
            # 卷积层：输入1通道(灰度图)，输出1通道，3x3卷积核
            self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
            # 全连接层：输入36个特征(8x8图像经过3x3卷积后变为6x6=36)，输出类别数
            self.fc = nn.Linear(36, cls_num)

        def forward(self, x):
            """
            前向传播
            Args:
                x: 输入图像张量
            Returns:
                out: 分类输出
            """
            x = self.convolution_layer(x)  # 卷积操作
            x = x.view(x.size(0), -1)      # 展平特征图为一维向量
            out = self.fc(x)               # 全连接层分类
            return out

    # 创建模型实例
    model = TinnyCNN(2)

    # step 3/4 : 优化模块
    # 定义损失函数：交叉熵损失，适用于多分类问题
    loss_f = nn.CrossEntropyLoss()
    
    # 定义优化器：随机梯度下降，设置学习率、动量和权重衰减
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 定义学习率调度器：每50个epoch将学习率乘以0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    
    # step 4/4 : 迭代模块
    # 开始训练循环，最多训练100个epoch
    for epoch in range(100):
        # 训练阶段
        model.train()  # 设置为训练模式，启用dropout和batch normalization
        for data, labels in train_loader:
            # 前向传播：计算模型输出
            outputs = model(data)
            
            # 清空梯度：每次反向传播前需要清空之前的梯度
            optimizer.zero_grad()

            # 计算损失：比较模型输出和真实标签
            loss = loss_f(outputs, labels)
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            correct_num = (predicted == labels).sum()  # 计算正确预测的数量
            acc = correct_num / labels.shape[0]        # 计算准确率
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))

        # 验证阶段
        model.eval()  # 设置为评估模式，关闭dropout和batch normalization
        for data, label in valid_loader:
            # 前向传播：计算模型输出
            outputs = model(data)

            # 计算验证损失
            loss = loss_f(outputs, labels)

            # 计算验证准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            correct_num = (predicted == labels).sum()  # 计算正确预测的数量
            acc_valid = correct_num / labels.shape[0]  # 计算准确率
            print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))

        # 早停条件：如果验证集准确率达到100%，则停止训练
        if acc_valid == 1:
            break

        # 学习率调整：根据调度器策略调整学习率
        scheduler.step()


if __name__ == "__main__":
    main()

