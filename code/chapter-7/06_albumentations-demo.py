# -*- coding:utf-8 -*-
"""
@file name  : 06_albumentations-demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : albumentations库学习 - 演示如何使用albumentations库进行图像数据增强
             
             albumentations是一个高性能的图像数据增强库，专门为计算机视觉任务设计
             相比torchvision.transforms，它具有更快的速度和更丰富的变换类型
             特别适合深度学习训练中的数据增强需求
"""
import cv2
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # =============================== 导入albumentations变换函数 ===============================
    from albumentations import (
        # 基础几何变换
        HorizontalFlip,      # 水平翻转
        Resize,              # 尺寸调整
        IAAPerspective,      # 透视变换（IAA增强版）
        ShiftScaleRotate,    # 平移、缩放、旋转的组合变换
        
        # 图像增强
        CLAHE,               # 对比度限制自适应直方图均衡化
        RandomRotate90,      # 随机90度旋转
        
        # 更多几何变换
        Transpose,           # 转置（行列交换）
        ShiftScaleRotate,    # 重复导入（可能是代码错误）
        Blur,                # 模糊效果
        
        # 扭曲和变形
        OpticalDistortion,   # 光学畸变
        GridDistortion,      # 网格畸变
        
        # 颜色和亮度调整
        HueSaturationValue,  # 色调、饱和度、明度调整
        
        # 噪声和模糊
        IAAAdditiveGaussianNoise,  # IAA高斯噪声
        GaussNoise,                 # 高斯噪声
        MotionBlur,                 # 运动模糊
        MedianBlur,                 # 中值模糊
        
        # 高级变换
        IAAPiecewiseAffine,  # IAA分段仿射变换
        IAASharpen,          # IAA锐化
        IAAEmboss,           # IAA浮雕效果
        
        # 对比度和亮度
        RandomContrast,      # 随机对比度调整
        RandomBrightness,    # 随机亮度调整
        
        # 组合和选择
        Flip,                # 翻转（通用）
        OneOf,               # 从多个变换中随机选择一个
        Compose              # 组合多个变换
    )  # 图像变换函数集合

    # =============================== 图像加载和预处理 ===============================
    # 设置图像文件路径
    path_img = r"F:\pytorch-tutorial-2nd\data\imgs\lena.png"
    
    # 使用OpenCV加载图像
    # cv2.imread(path_img, 1) 中的1表示以彩色模式加载（BGR格式）
    # 返回值是numpy数组，形状为(height, width, 3)
    image = cv2.imread(path_img, 1)  # BGR格式
    
    # 将BGR格式转换为RGB格式
    # OpenCV默认使用BGR格式，而matplotlib使用RGB格式
    # 这个转换确保图像在matplotlib中正确显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =============================== 应用图像变换 ===============================
    
    # 1. 尺寸调整变换
    # Resize是DualTransform类型，可以同时处理图像和标签
    # width=224, height=224: 将图像调整为224x224像素
    resize = Resize(width=224, height=224)  # DualTransform类型
    
    # 应用尺寸调整变换
    # resize(image=image)返回字典，包含变换后的图像
    # 使用['image']键获取变换后的图像
    img_HorizontalFlip = resize(image=image)['image']
    
    # 2. 模糊变换
    # Blur是ImageOnlyTransform类型，只处理图像，不处理标签
    # p=1: 概率为1，表示100%应用此变换
    blur = Blur(p=1)  # ImageOnlyTransform类型
    
    # 应用模糊变换
    # 注意：这里变量名可能有些混淆，实际应用的是模糊效果
    img_ShiftScaleRotate = blur(image=image)['image']
    
    # 3. 随机90度旋转
    # RandomRotate90: 随机选择0°, 90°, 180°, 270°中的一个角度进行旋转
    # p=1: 概率为1，表示100%应用此变换
    rotate90 = RandomRotate90(p=1)
    
    # 应用随机旋转变换
    img_RandomRotate90 = rotate90(image=image)['image']

    # =============================== 可视化变换结果 ===============================
    # 创建2x2的子图布局，显示原始图像和三种变换后的图像
    
    # 子图1: 原始图像
    plt.subplot(221).imshow(image)
    plt.title("raw img")  # 设置标题为"原始图像"
    
    # 子图2: 尺寸调整后的图像
    # 注意：这里显示的是resize变换的结果，但变量名是img_HorizontalFlip
    plt.subplot(222).imshow(img_HorizontalFlip)
    
    # 子图3: 模糊处理后的图像
    # 注意：这里显示的是blur变换的结果，但变量名是img_ShiftScaleRotate
    plt.subplot(223).imshow(img_ShiftScaleRotate)
    
    # 子图4: 随机旋转后的图像
    plt.subplot(224).imshow(img_RandomRotate90)
    
    # 显示所有子图
    plt.show()




