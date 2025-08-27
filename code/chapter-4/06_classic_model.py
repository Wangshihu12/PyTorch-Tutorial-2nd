# -*- coding:utf-8 -*-
"""
@file name  : 06_classic_model.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-13
@brief      : torchvision 经典模型学习
@description: 演示如何使用torchvision.models加载预定义的经典深度学习模型
             包括AlexNet、VGG16、GoogLeNet和ResNet50等经典架构
             展示如何对模型权重进行自定义初始化
"""
import torch
import torch.nn as nn
from torchvision import models

# 加载预定义的经典深度学习模型
# 这些模型都经过ImageNet数据集预训练，具有强大的特征提取能力

# 创建AlexNet模型实例
# AlexNet是2012年ImageNet竞赛的冠军，首次证明了深度CNN的有效性
model_alexnet = models.alexnet()

# 创建VGG16模型实例
# VGG16是2014年ImageNet竞赛的亚军，以其简洁的架构设计著称
# 使用3x3卷积核和2x2池化层的堆叠结构
model_vgg16 = models.vgg16()

# 创建GoogLeNet模型实例
# GoogLeNet是2014年ImageNet竞赛的冠军，引入了Inception模块
# 通过多尺度特征提取和1x1卷积降维，显著减少了参数量
model_googlenet = models.googlenet()

# 创建ResNet50模型实例
# ResNet是2015年ImageNet竞赛的冠军，解决了深度网络的梯度消失问题
# 通过残差连接（skip connection）实现了超深网络的训练
model_resnet50 = models.resnet50()


# 对AlexNet模型进行权重初始化
# 遍历模型中的所有模块，找到卷积层并进行权重初始化
for m in model_alexnet.modules():
    # 检查当前模块是否为2D卷积层
    if isinstance(m, torch.nn.Conv2d):
        # 使用Kaiming初始化方法初始化卷积层权重
        # Kaiming初始化专门为使用ReLU激活函数的深度网络设计
        # mode='fan_out': 保持前向传播中方差不变
        # nonlinearity='relu': 针对ReLU激活函数优化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
