# -*- coding:utf-8 -*-
"""
@file name  : 02_summarywriter.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-07
@brief      : tensorboard的writer 使用
@description: 演示PyTorch中SummaryWriter的各种功能，用于TensorBoard可视化
             包括标量数据、图像数据、直方图、3D网格和超参数等不同类型数据的记录
             展示如何将训练过程中的各种信息可视化到TensorBoard中
"""
import os
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # 主程序：演示SummaryWriter的各种功能

    # ================================ add_scalar演示 ================================
    # 记录单个标量值，用于绘制曲线图（如损失曲线、准确率曲线等）
    
    # 创建SummaryWriter实例，添加注释和文件名后缀
    writer = SummaryWriter(comment="add_scalar", filename_suffix="_test_tensorboard")
    
    x = range(100)  # 创建100个数据点
    for i in x:
        # 记录数学函数y=2x的值
        writer.add_scalar('y=2x', i * 2, i)
        # 记录训练损失（随机值模拟）
        writer.add_scalar('Loss/train', np.random.random(), i)
        # 记录验证损失（随机值模拟）
        writer.add_scalar('Loss/Valid', np.random.random(), i)
    
    writer.close()  # 关闭writer，确保数据写入磁盘

    # ================================ add_scalars演示 ================================
    # 同时记录多个标量值，用于在同一图表中比较多条曲线
    
    writer = SummaryWriter(comment="add_scalars", filename_suffix="_test_tensorboard")
    for i in range(100):
        # 在同一个图表中记录训练损失和验证损失
        writer.add_scalars('Loss_curve', {
            'train_loss': np.random.random(),    # 训练损失
            'valid_loss': np.random.random()     # 验证损失
        }, i)
    writer.close()

    # ================================ add_histogram演示 ================================
    # 记录数据的分布直方图，用于分析参数分布、梯度分布等
    
    writer = SummaryWriter(comment="add_histogram", filename_suffix="_test_tensorboard")
    for i in range(10):
        x = np.random.random(1000)  # 生成1000个随机数
        # 记录分布中心，每次迭代时分布会向右移动i个单位
        writer.add_histogram('distribution centers', x + i, i)
    writer.close()

    # ================================ add_image演示 ================================
    # 记录单张图像，支持CHW和HWC两种格式
    
    writer = SummaryWriter(comment="add_image", filename_suffix="_test_tensorboard")

    # 创建CHW格式的图像（通道在前）：3通道，100x100像素
    img = np.zeros((3, 100, 100))
    # 第一个通道：从0到1的渐变
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # 第二个通道：从1到0的渐变（与第一个通道相反）
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    # 创建HWC格式的图像（通道在后）：100x100像素，3通道
    img_HWC = np.zeros((100, 100, 3))
    # 第一个通道：从0到1的渐变
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # 第二个通道：从1到0的渐变
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    # 记录CHW格式的图像
    writer.add_image('my_image-shape:{}'.format(img.shape), img, 0)
    print(img.shape)  # 打印图像形状：(3, 100, 100)

    # 记录HWC格式的图像，需要指定dataformats参数
    writer.add_image('my_image_HWC-shape:{}'.format(img_HWC.shape), img_HWC, 0, dataformats='HWC')
    print(img_HWC.shape)  # 打印图像形状：(100, 100, 3)

    # 数据集下载链接：
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    
    # 读取真实的COVID-19医学图像
    path_img = r"/home/wang/github/PyTorch-Tutorial-2nd/data/covid-19-dataset2&3/covid-19-dataset-3/imgs/ryct.2020200028.fig1a.jpeg"
    img_opencv = cv2.imread(path_img)  # OpenCV读取图像，默认BGR格式
    # 记录OpenCV读取的图像，指定HWC格式
    writer.add_image('img_opencv_HWC-shape:{}'.format(img_opencv.shape), img_opencv, 0, dataformats='HWC')
    writer.close()

    # ================================ add_images演示 ================================
    # 记录图像批次，用于可视化多个样本
    
    writer = SummaryWriter(comment="add_images", filename_suffix="_test_tensorboard")

    # 创建图像批次：16张图像，每张3通道，100x100像素
    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        # 第一通道：渐变强度随批次索引变化
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        # 第二通道：反向渐变强度随批次索引变化
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

    # 记录图像批次
    writer.add_images('add_images', img_batch, 0)
    writer.close()

    # ================================ add_mesh演示 ================================
    # 记录3D网格数据，用于可视化3D模型
    
    import torch
    
    # 定义四面体的顶点坐标（4个顶点，每个顶点3个坐标）
    vertices_tensor = torch.as_tensor([
        [1, 1, 1],      # 顶点0：(1, 1, 1)
        [-1, -1, 1],    # 顶点1：(-1, -1, 1)
        [1, -1, -1],    # 顶点2：(1, -1, -1)
        [-1, 1, -1],    # 顶点3：(-1, 1, -1)
    ], dtype=torch.float).unsqueeze(0)  # 添加批次维度
    
    # 定义每个顶点的颜色（RGB格式）
    colors_tensor = torch.as_tensor([
        [255, 0, 0],     # 顶点0：红色
        [0, 255, 0],     # 顶点1：绿色
        [0, 0, 255],     # 顶点2：蓝色
        [255, 0, 255],   # 顶点3：紫色
    ], dtype=torch.int).unsqueeze(0)  # 添加批次维度
    
    # 定义面片（三角形）：每个面由3个顶点索引组成
    faces_tensor = torch.as_tensor([
        [0, 2, 3],  # 面0：由顶点0、2、3组成
        [0, 3, 1],  # 面1：由顶点0、3、1组成
        [0, 1, 2],  # 面2：由顶点0、1、2组成
        [1, 3, 2],  # 面3：由顶点1、3、2组成
    ], dtype=torch.int).unsqueeze(0)  # 添加批次维度

    # 创建SummaryWriter并记录3D网格
    writer = SummaryWriter(comment="add_mesh", filename_suffix="_test_tensorboard")
    writer.add_mesh('add_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)
    writer.close()

    # ================================ add_hparams演示 ================================
    # 记录超参数和对应的性能指标，用于超参数调优分析
    
    writer = SummaryWriter(comment="add_hparams", filename_suffix="_test_tensorboard")
    for i in range(5):
        # 记录超参数：学习率和批次大小
        hparams = {'lr': 0.1 * i, 'bsize': i}
        # 记录对应的性能指标：准确率和损失
        metrics = {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i}
        # 将超参数和性能指标关联记录
        writer.add_hparams(hparams, metrics)
    writer.close()
