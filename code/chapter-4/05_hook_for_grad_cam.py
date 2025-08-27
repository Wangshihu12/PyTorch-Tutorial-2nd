# -*- coding:utf-8 -*-
"""
@file name  : 05_hook_for_grad_cam.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-12
@brief      : 通过实现Grad-CAM学习module中的forward_hook和full_backward_hook函数
@description: 实现Grad-CAM（Gradient-weighted Class Activation Mapping）算法
             通过PyTorch的hook机制获取中间层特征图和梯度信息
             生成类激活图，可视化模型关注的图像区域
"""

import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net(nn.Module):
    """
    简单的卷积神经网络模型，用于CIFAR-10图像分类
    网络结构：两个卷积块 + 三个全连接层
    """
    
    def __init__(self):
        """
        初始化网络结构
        """
        super(Net, self).__init__()
        # 第一个卷积块：3通道输入，6通道输出，5x5卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 最大池化层：2x2池化窗口，步长为2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积块：6通道输入，16通道输出，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 最大池化层：2x2池化窗口，步长为2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层：展平后的特征图到120个神经元
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层：120到84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层：84到10个类别（CIFAR-10）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播
        
        @param x: 输入张量，形状为(B, 3, 32, 32)
        @return: 输出张量，形状为(B, 10)，表示10个类别的概率分布
        """
        # 第一个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool1(F.relu(self.conv1(x)))
        # 第二个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool1(F.relu(self.conv2(x)))
        # 展平特征图：从(B, 16, 5, 5)到(B, 16*5*5)
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层：线性变换 -> ReLU激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层：线性变换，不激活
        x = self.fc3(x)
        return x


def img_transform(img_in, transform):
    """
    将图像进行预处理，并转换成模型输入所需的形式
    
    @param img_in: 输入的numpy数组图像
    @param transform: torchvision.transforms变换对象
    @return: 预处理后的PyTorch张量，形状为(1, C, H, W)
    """
    img = img_in.copy()  # 复制输入图像，避免修改原始数据
    img = Image.fromarray(np.uint8(img))  # 转换为PIL图像
    img = transform(img)  # 应用预处理变换
    img = img.unsqueeze(0)    # 添加批次维度：C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    
    @param img_in: 输入的numpy数组，形状为[H, W, C]
    @return: 预处理后的图像张量，形状为(1, C, H, W)
    """
    img = img_in.copy()  # 复制输入图像
    img = cv2.resize(img, (32, 32))  # 调整图像大小为32x32
    img = img[:, :, ::-1]   # BGR --> RGB：OpenCV默认BGR，转换为RGB
    # 定义预处理变换：转换为张量 + 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为PyTorch张量，范围0-1
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974],  # RGB三通道均值
                           [0.24580306, 0.24236229, 0.2603115])    # RGB三通道标准差
    ])
    img_input = img_transform(img, transform)  # 应用预处理
    return img_input


def backward_hook(module, grad_in, grad_out):
    """
    反向传播钩子函数，用于捕获梯度信息
    
    @param module: 注册hook的模块
    @param grad_in: 输入梯度（通常不使用）
    @param grad_out: 输出梯度，包含我们需要的梯度信息
    """
    # 将梯度信息存储到全局列表中，用于后续的Grad-CAM计算
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    """
    前向传播钩子函数，用于捕获特征图信息
    
    @param module: 注册hook的模块
    @param input: 模块的输入
    @param output: 模块的输出，即特征图
    """
    # 将特征图信息存储到全局列表中，用于后续的Grad-CAM计算
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir):
    """
    将CAM热力图叠加到原始图像上，并保存结果
    
    @param img: 原始图像，numpy数组
    @param mask: CAM热力图，numpy数组
    @param out_dir: 输出目录路径
    """
    # 将CAM热力图转换为彩色热力图（JET颜色映射）
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255  # 归一化到0-1范围
    
    # 将热力图叠加到原始图像上
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)  # 归一化到0-1范围

    # 设置输出文件路径
    path_cam_img = os.path.join(out_dir, "cam.jpg")      # CAM叠加图像
    path_raw_img = os.path.join(out_dir, "raw.jpg")      # 原始图像
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 保存图像：将0-1范围转换为0-255范围
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    print("save img to :{}".format(path_cam_img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算指定类别的类向量，用于反向传播
    
    @param ouput_vec: 模型输出的张量
    @param index: 指定类别索引，如果为None则选择预测概率最高的类别
    @return: 类向量张量，用于计算梯度
    """
    if not index:
        # 如果没有指定类别，选择预测概率最高的类别
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    
    # 扩展索引维度，用于创建one-hot向量
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    
    # 创建one-hot向量：在指定位置设置为1，其他位置为0
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True  # 设置requires_grad=True，用于反向传播
    
    # 计算类向量：one-hot向量与模型输出的点积
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    根据梯度和特征图生成CAM（Class Activation Map）
    
    @param feature_map: 特征图，numpy数组，形状为[C, H, W]
    @param grads: 梯度信息，numpy数组，形状为[C, H, W]
    @return: CAM热力图，numpy数组，形状为[H, W]
    """
    # 初始化CAM：形状为特征图的空间维度(H, W)
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    # 计算权重：对每个通道的梯度在空间维度上求平均
    weights = np.mean(grads, axis=(1, 2))  # 形状为[C]

    # 加权求和：将每个通道的特征图乘以对应的权重
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    # 后处理：确保CAM值非负，并归一化到0-1范围
    cam = np.maximum(cam, 0)  # ReLU操作：将负值置为0
    cam = cv2.resize(cam, (32, 32))  # 调整大小到32x32
    cam -= np.min(cam)  # 减去最小值
    cam /= np.max(cam)  # 除以最大值，归一化到0-1

    return cam


if __name__ == '__main__':
    # 主程序：演示Grad-CAM的完整流程

    # 设置基础路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 数据集下载链接：https://pan.baidu.com/s/1eDwZchwp6P1Ab9d8Qn6rbA  提取码：l8qe
    path_img = os.path.join(BASE_DIR, "grad_cam_data", "cam_img", "test_img_8.png")  # 测试图像路径
    path_net = os.path.join(BASE_DIR, "grad_cam_data", "net_params_72p.pkl")         # 预训练模型路径
    output_dir = os.path.join(BASE_DIR, "grad_cam_data", "results", "backward_hook_cam")  # 输出目录

    # CIFAR-10数据集的10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 全局列表：用于存储hook捕获的特征图和梯度信息
    fmap_block = list()  # 存储特征图
    grad_block = list()  # 存储梯度信息

    # 第一步：图片读取和网络加载
    img = cv2.imread(path_img, 1)  # 读取图像，H*W*C格式
    img_input = img_preprocess(img)  # 预处理图像
    net = Net()  # 创建网络实例
    net.load_state_dict(torch.load(path_net))  # 加载预训练权重

    # 第二步：注册hook函数
    # 在conv2层注册前向hook，用于捕获特征图
    net.conv2.register_forward_hook(farward_hook)
    # 在conv2层注册反向hook，用于捕获梯度信息
    net.conv2.register_full_backward_hook(backward_hook)

    # 第三步：前向传播
    output = net(img_input)  # 输入图像，获得模型输出
    idx = np.argmax(output.cpu().data.numpy())  # 获取预测类别索引
    print("predict: {}".format(classes[idx]))  # 打印预测结果

    # 第四步：反向传播
    net.zero_grad()  # 清零梯度
    class_loss = comp_class_vec(output)  # 计算指定类别的类向量
    class_loss.backward()  # 反向传播，计算梯度

    # 第五步：生成CAM
    # 从hook中获取梯度和特征图信息
    grads_val = grad_block[0].cpu().data.numpy().squeeze()  # 梯度信息
    fmap = fmap_block[0].cpu().data.numpy().squeeze()       # 特征图信息
    cam = gen_cam(fmap, grads_val)  # 生成CAM热力图

    # 第六步：保存CAM图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255  # 调整图像大小并归一化
    show_cam_on_image(img_show, cam, output_dir)  # 将CAM叠加到图像上并保存
















