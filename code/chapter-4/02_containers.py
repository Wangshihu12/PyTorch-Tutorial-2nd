# -*- coding:utf-8 -*-
"""
@file name  : 02_containers.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-30
@brief      : 熟悉常用容器：sequential, modulelist
@description: 演示PyTorch中三种重要的神经网络容器类的使用方法
             包括Sequential、ModuleList和ModuleDict，这些容器用于组织和管理神经网络层
"""
import torch
import torch.nn as nn
from torchvision.models import alexnet


if __name__ == "__main__":
    # ========================== Sequential容器演示 ==========================
    # Sequential是最简单的容器，按顺序执行多个层
    # 适用于简单的线性堆叠网络结构
    
    # 加载预训练的AlexNet模型（不包含预训练权重）
    model = alexnet(pretrained=False)
    
    # 创建模拟输入数据：批次大小为1，3个通道，224x224像素的图像
    fake_input = torch.randn((1, 3, 224, 224))
    
    # 前向传播，通过整个网络
    output = model(fake_input)

    # ========================== ModuleList容器演示 ==========================
    # ModuleList用于存储多个模块，可以动态访问和修改
    # 与普通Python列表的区别：ModuleList中的模块会被正确注册到模型的参数中

    class MyModule(nn.Module):
        """
        演示ModuleList使用的自定义模块类
        包含10个线性层，每个层的输入输出维度都是10
        """
        
        def __init__(self):
            """
            初始化模块，创建10个线性层
            """
            super(MyModule, self).__init__()
            # 使用ModuleList创建10个线性层，每个层输入输出维度都是10
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            
            # 注意：如果使用普通Python列表，这些层不会被注册到模型的参数中
            # self.linears = [nn.Linear(10, 10) for i in range(10)]    
            # 观察model._modules，将会是空的，导致参数无法被正确管理

        def forward(self, x):
            """
            前向传播：依次通过所有线性层
            
            @param x: 输入张量，形状为(batch_size, 10)
            @return: 经过所有线性层处理后的输出张量
            """
            for sub_layer in self.linears:
                x = sub_layer(x)  # 依次通过每个线性层
            return x

    # 创建MyModule实例
    model = MyModule()
    
    # 创建模拟输入：32个样本，每个样本10个特征
    fake_input = torch.randn((32, 10))
    
    # 前向传播
    output = model(fake_input)
    print(output.shape)  # 输出形状：(32, 10)

    # ========================== ModuleDict容器演示 ==========================
    # ModuleDict用于存储命名的模块，可以通过字符串键访问
    # 适用于需要根据条件选择不同网络结构的场景

    class MyModule2(nn.Module):
        """
        演示ModuleDict使用的自定义模块类
        包含可选择的卷积/池化层和激活函数
        """
        
        def __init__(self):
            """
            初始化模块，创建可选择的网络层和激活函数
            """
            super(MyModule2, self).__init__()
            
            # 创建可选择的网络层：卷积层或池化层
            self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(3, 16, 5),      # 3通道输入，16通道输出，5x5卷积核
                'pool': nn.MaxPool2d(3)            # 3x3最大池化层
            })
            
            # 创建可选择的激活函数
            self.activations = nn.ModuleDict({
                'lrelu': nn.LeakyReLU(),           # LeakyReLU激活函数
                'prelu': nn.PReLU()                # PReLU激活函数（参数化ReLU）
            })

        def forward(self, x, choice, act):
            """
            前向传播：根据选择执行相应的网络层和激活函数
            
            @param x: 输入张量
            @param choice: 选择的网络层类型，'conv'或'pool'
            @param act: 选择的激活函数类型，'lrelu'或'prelu'
            @return: 经过选择的网络层和激活函数处理后的输出
            """
            x = self.choices[choice](x)    # 根据choice选择执行卷积或池化
            x = self.activations[act](x)   # 根据act选择执行相应的激活函数
            return x

    # 创建MyModule2实例
    model2 = MyModule2()
    
    # 创建模拟输入：1个样本，3个通道，7x7像素的图像
    fake_input = torch.randn((1, 3, 7, 7))
    
    # 测试不同的网络层和激活函数组合
    convout = model2(fake_input, "conv", "lrelu")  # 使用卷积层+LeakyReLU
    poolout = model2(fake_input, "pool", "prelu")  # 使用池化层+PReLU
    
    # 打印输出形状
    print(convout.shape, poolout.shape)  # 显示两种不同组合的输出形状