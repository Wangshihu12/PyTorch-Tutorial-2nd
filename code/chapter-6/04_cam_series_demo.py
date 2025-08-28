# -*- coding:utf-8 -*-
"""
@file name  : 04_cam_series_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-156
@brief      : https://github.com/jacobgil/pytorch-grad-cam  学习与使用
安装：pip install grad_cam
"""
import cv2
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# 导入各种CAM（Class Activation Mapping）算法
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from matplotlib import pyplot as plt

def img_transform(img_in, transform):
    """
    将图像进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    
    该函数将PIL图像转换为PyTorch张量，并添加batch维度
    
    @param img_in: numpy.ndarray, 输入的图像数组
    @param transform: torchvision.transforms, 图像预处理变换
    @return: torch.Tensor, 形状为(1, C, H, W)的张量，即B*C*H*W格式
    """
    # 复制输入图像，避免修改原始数据
    img = img_in.copy()
    # 将numpy数组转换为PIL图像对象
    img = Image.fromarray(np.uint8(img))
    # 应用预处理变换（ToTensor, Normalize等）
    img = transform(img)
    # 添加batch维度：C*H*W --> B*C*H*W
    img = img.unsqueeze(0)
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    
    该函数完成图像的完整预处理流程，包括尺寸调整、颜色空间转换、标准化等
    
    @param img_in: numpy.ndarray, 输入的图像数组，形状为[H, W, C]
    @return: torch.Tensor, 预处理后的图像张量，形状为(1, C, H, W)
    """
    # 复制输入图像，避免修改原始数据
    img = img_in.copy()
    # 调整图像尺寸为224x224（ResNet的标准输入尺寸）
    img = cv2.resize(img, (224, 224))
    # 颜色空间转换：BGR --> RGB
    # OpenCV默认读取为BGR格式，需要转换为RGB格式
    img = img[:, :, ::-1]
    
    # 定义图像预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),        # 转换为PyTorch张量，并将像素值归一化到[0,1]
        # 使用ImageNet数据集的标准化参数
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974],  # RGB三个通道的均值
                           [0.24580306, 0.24236229, 0.2603115])   # RGB三个通道的标准差
    ])
    
    # 调用img_transform函数完成最终的预处理
    img_input = img_transform(img, transform)
    return img_input


def cam_factory(cam_name_):
    """
    CAM算法工厂函数
    
    根据算法名称动态创建对应的CAM实例
    
    @param cam_name_: str, CAM算法的名称字符串
    @return: CAM算法实例
    """
    return eval(cam_name_)


if __name__ == '__main__':
    # =============================== 主程序：CAM系列算法演示 ===============================
    
    # 设置输入图像路径和输出目录
    path_img = "both.png"        # 输入图像路径
    output_dir = "./Result"      # 结果输出目录

    # =============================== 图像预处理 ===============================
    # 使用OpenCV读取图像
    # cv2.imread(path_img, 1) 中的1表示读取为彩色图像
    # 返回的图像格式为H*W*C（高度、宽度、通道）
    img = cv2.imread(path_img, 1)
    
    # 对图像进行预处理，转换为模型输入格式
    img_input = img_preprocess(img)

    # =============================== 模型加载 ===============================
    # 加载预训练的ResNet50模型
    model = resnet50(pretrained=True)
    
    # 设置目标层：选择模型的最后一个卷积层作为CAM分析的目标
    # model.layer4[-1] 表示ResNet的最后一个残差块的最后一个卷积层
    target_layers = [model.layer4[-1]]
    
    # 设置输入张量
    input_tensor = img_input

    # =============================== CAM算法列表 ===============================
    # 定义要演示的CAM算法列表
    # 这些算法都是不同的类激活映射方法，用于可视化模型的决策过程
    cam_alg_list = "GradCAM,ScoreCAM,GradCAMPlusPlus,XGradCAM,EigenCAM,FullGrad".split(",")
    
    # 算法说明：
    # - GradCAM: 基于梯度的类激活映射
    # - ScoreCAM: 基于分数的类激活映射，不需要梯度
    # - GradCAMPlusPlus: GradCAM的改进版本
    # - XGradCAM: 基于期望梯度的类激活映射
    # - EigenCAM: 基于特征图主成分分析的类激活映射
    # - FullGrad: 基于完整梯度的类激活映射

    # =============================== 可视化设置 ===============================
    # 设置matplotlib布局，确保子图之间有适当的间距
    plt.tight_layout()
    
    # 创建2行3列的子图布局，用于显示6种不同的CAM结果
    # fig, axs = plt.subplots(2, 3, figsize=(9, 9))  # 可以设置图像大小
    fig, axs = plt.subplots(2, 3)
    
    # =============================== CAM算法演示循环 ===============================
    # 遍历每种CAM算法
    for idx, cam_name in enumerate(cam_alg_list):
        # 使用工厂函数创建CAM算法实例
        cam = cam_factory(cam_name)(model=model, target_layers=target_layers)
        
        # 生成类激活映射
        # targets=None: 如果不指定目标类别，则自动选择得分最高的类别
        # 也可以指定特定类别：targets = [ClassifierOutputTarget(281)]  # 例如类别281
        
        # 可选参数：
        # aug_smooth=True: 应用测试时数据增强平滑
        # eigen_smooth=True: 应用特征图主成分分析平滑
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        # 提取第一个（也是唯一一个）图像的CAM结果
        # 因为我们的输入只有一张图像，所以取[0, :]
        grayscale_cam = grayscale_cam[0, :]
        
        # 图像归一化：将像素值从[0,255]范围归一化到[0,1]范围
        # 方法1：使用OpenCV的normalize函数
        # img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        # 方法2：直接除以255（更简单）
        img_norm = img/255.
        
        # 将CAM热力图叠加到原始图像上
        # use_rgb=False: 因为我们的图像是BGR格式
        visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=False)
        
        # 将BGR格式转换为RGB格式，以便matplotlib正确显示
        vis_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)

        # 在对应的子图中显示结果
        im = axs.ravel()[idx].imshow(vis_rgb)
        # 设置子图标题为算法名称
        axs.ravel()[idx].set_title(cam_name)
    
    # 显示所有子图
    plt.show()