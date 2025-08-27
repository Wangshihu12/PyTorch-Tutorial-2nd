# -*- coding:utf-8 -*-
"""
@file name  : 03_lr_scheduler.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-01
@brief      : lr scheduelr 学习
@description: 演示完整的深度学习模型训练流程，重点展示学习率调度器的使用
             包括数据加载、模型构建、优化器设置、训练循环和验证过程
             展示如何在实际训练中应用学习率调度策略
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
    """
    主函数：演示完整的深度学习模型训练流程，重点展示学习率调度器的使用
    
    训练流程分为四个主要步骤：
    1. 数据模块：构建dataset和dataloader，实现数据读取和预处理
    2. 模型模块：构建神经网络架构
    3. 优化模块：设置损失函数、优化器和学习率调度器
    4. 迭代模块：执行训练循环，包括训练和验证，并动态调整学习率
    """
    
    # 思考：如何实现你的模型训练？第一步干什么？第二步干什么？...第n步...
    # step 1/4 : 数据模块：构建dataset, dataloader，实现对硬盘中数据的读取及设定预处理方法
    # step 2/4 : 模型模块：构建神经网络，用于后续训练
    # step 3/4 : 优化模块：设定损失函数与优化器，用于在训练过程中对网络参数进行更新
    # step 4/4 : 迭代模块: 循环迭代地进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到我们让它停止训练

    # ========================== step 1/4 : 数据模块 ==========================
    # 构建自定义数据集类，用于加载COVID-19医学图像数据
    
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
    
    # 数据集下载链接：https://pan.baidu.com/s/18BsxploWR3pbybFtNsw5fA  提取码：pyto
    
    # 设置数据集路径
    root_dir = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-demo"  # COVID-19数据集根目录
    img_dir = os.path.join(root_dir, "imgs")  # 图像文件目录
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")  # 训练标签文件路径
    path_txt_valid = os.path.join(root_dir, "labels", "valid.txt")  # 验证标签文件路径
    
    # 定义图像预处理变换：调整大小到8x8并转换为张量
    transforms_func = transforms.Compose([
        transforms.Resize((8, 8)),  # 将图像调整为8x8大小
        transforms.ToTensor(),      # 转换为PyTorch张量
    ])
    
    # 创建训练和验证数据集
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    
    # 创建数据加载器，批次大小为2
    train_loader = DataLoader(dataset=train_data, batch_size=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=2)

    # ========================== step 2/4 : 模型模块 ==========================
    # 构建一个简单的卷积神经网络模型
    
    class TinnyCNN(nn.Module):
        """
        简单的CNN模型类，用于COVID-19图像分类
        网络结构：1个卷积层 + 1个全连接层
        """
        
        def __init__(self, cls_num=2):
            """
            初始化网络结构
            
            @param cls_num: 分类类别数，默认为2（COVID-19二分类）
            """
            super(TinnyCNN, self).__init__()
            # 卷积层：1通道输入，1通道输出，3x3卷积核
            self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
            # 全连接层：输入特征数36，输出类别数cls_num
            # 36 = (8-3+1) * (8-3+1) = 6 * 6，考虑了卷积后的特征图大小
            self.fc = nn.Linear(36, cls_num)

        def forward(self, x):
            """
            前向传播
            
            @param x: 输入张量，形状为(B, 1, 8, 8)
            @return: 输出张量，形状为(B, cls_num)
            """
            x = self.convolution_layer(x)  # 卷积操作
            x = x.view(x.size(0), -1)      # 展平特征图：从(B, 1, 6, 6)到(B, 36)
            out = self.fc(x)               # 全连接层分类
            return out

    # 创建模型实例：二分类任务
    model = TinnyCNN(2)

    # ========================== step 3/4 : 优化模块 ==========================
    # 设置损失函数、优化器和学习率调度器
    
    # 损失函数：交叉熵损失，适用于多分类问题
    loss_f = nn.CrossEntropyLoss()
    
    # 优化器：SGD（随机梯度下降）
    # lr=0.1: 学习率
    # momentum=0.9: 动量参数，帮助加速收敛
    # weight_decay=5e-4: 权重衰减（L2正则化），防止过拟合
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 学习率调度器：StepLR，每50个epoch将学习率乘以0.1
    # gamma=0.1: 学习率衰减因子
    # step_size=50: 每50个epoch调整一次学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    
    # ========================== step 4/4 : 迭代模块 ==========================
    # 执行训练循环，包括训练和验证过程，并动态调整学习率
    
    for epoch in range(100):  # 训练100个epoch
        # ========================== 训练阶段 ==========================
        # 设置模型为训练模式（启用dropout、batch normalization等）
        model.train()
        
        # 遍历训练数据批次
        for data, labels in train_loader:
            # 前向传播：输入数据，获得模型输出
            outputs = model(data)
            
            # 清零梯度：防止梯度累积
            optimizer.zero_grad()

            # 计算损失：比较模型输出和真实标签
            loss = loss_f(outputs, labels)
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 计算训练集分类准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别（最大概率的类别）
            correct_num = (predicted == labels).sum()  # 计算正确预测的样本数
            acc = correct_num / labels.shape[0]        # 计算准确率
            
            # 打印训练信息：当前epoch、损失值和准确率
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
            # print(predicted, labels)  # 可选：打印预测结果和真实标签

        # ========================== 验证阶段 ==========================
        # 设置模型为评估模式（禁用dropout、batch normalization等）
        model.eval()
        
        # 遍历验证数据批次
        for data, label in valid_loader:
            # 前向传播：输入数据，获得模型输出
            outputs = model(data)

            # 计算验证集损失：比较模型输出和真实标签
            loss = loss_f(outputs, labels)

            # 计算验证集分类准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            correct_num = (predicted == labels).sum()  # 计算正确预测的样本数
            acc_valid = correct_num / labels.shape[0]  # 计算验证准确率
            
            # 打印验证信息：当前epoch、损失值和准确率
            print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))

        # ========================== 早停和学习率调整 ==========================
        # 添加早停条件：当验证准确率达到100%时停止训练
        if acc_valid == 1:
            print("验证准确率达到100%，提前停止训练！")
            break

        # 学习率调整：每个epoch结束后调用调度器
        # StepLR会在每50个epoch后将学习率乘以0.1
        scheduler.step()


if __name__ == "__main__":
    main()  # 执行主函数

