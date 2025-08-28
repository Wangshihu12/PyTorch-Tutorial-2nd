# -*- coding:utf-8 -*-
"""
@file name  : 04_grad_cam_pp.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-15
@brief      : Grad-CAM++ 演示
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


def load_class_names(p_clsnames):
    """
    加载ImageNet数据集的类别名称
    
    从JSON文件中读取1000个ImageNet类别的名称，用于显示预测结果
    
    @param p_clsnames: str, JSON文件路径，包含类别名称的映射
    @return: dict, 类别索引到类别名称的映射字典
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    return class_names


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


def backward_hook(module, grad_in, grad_out):
    """
    反向传播Hook函数
    
    在反向传播过程中捕获指定层的梯度信息，用于Grad-CAM计算
    
    @param module: torch.nn.Module, 注册Hook的模块
    @param grad_in: tuple, 输入梯度（通常不使用）
    @param grad_out: tuple, 输出梯度，我们需要的梯度信息
    """
    # 将梯度信息添加到全局列表中，detach()分离梯度避免内存泄漏
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    """
    前向传播Hook函数
    
    在前向传播过程中捕获指定层的特征图输出，用于Grad-CAM计算
    
    @param module: torch.nn.Module, 注册Hook的模块
    @param input: tuple, 模块的输入（通常不使用）
    @param output: torch.Tensor, 模块的输出特征图
    """
    # 将特征图信息添加到全局列表中
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir):
    """
    将CAM热力图叠加到原始图像上并保存
    
    该函数将生成的CAM热力图与原始图像融合，生成可视化结果
    
    @param img: numpy.ndarray, 原始图像，像素值范围[0,1]
    @param mask: numpy.ndarray, CAM热力图，像素值范围[0,1]
    @param out_dir: str, 输出目录路径
    """
    # 将CAM热力图转换为彩色热力图（JET颜色映射）
    # 将[0,1]范围的mask转换为[0,255]范围，然后应用JET颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    # 将热力图归一化到[0,1]范围
    heatmap = np.float32(heatmap) / 255
    
    # 将热力图与原始图像叠加
    cam = heatmap + np.float32(img)
    # 归一化叠加结果到[0,1]范围
    cam = cam / np.max(cam)

    # 设置输出文件路径
    path_cam_img = os.path.join(out_dir, "cam.jpg")      # CAM叠加结果
    path_raw_img = os.path.join(out_dir, "raw.jpg")      # 原始图像
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 保存图像，将[0,1]范围转换为[0,255]范围
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))      # 保存CAM叠加图像
    cv2.imwrite(path_raw_img, np.uint8(255 * img))      # 保存原始图像


def comp_class_vec(ouput_vec, index=None):
    """
    计算类别向量，用于反向传播
    
    该函数创建一个one-hot向量，指定要计算梯度的目标类别
    
    @param ouput_vec: torch.Tensor, 模型的输出向量，形状为(1, 1000)
    @param index: int, 指定类别索引，如果为None则选择预测概率最高的类别
    @return: torch.Tensor, 目标类别的损失值，用于反向传播
    """
    if not index:
        # 如果没有指定类别，选择预测概率最高的类别
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        # 如果指定了类别，转换为numpy数组
        index = np.array(index)
    
    # 添加维度，将index转换为(1, 1)形状
    index = index[np.newaxis, np.newaxis]
    # 转换为PyTorch张量
    index = torch.from_numpy(index)
    
    # 创建one-hot向量，形状为(1, 1000)
    # scatter_函数在指定位置设置值为1
    one_hot = torch.zeros(1, 1000).scatter_(1, index, 1)
    # 设置requires_grad=True，使该张量参与梯度计算
    one_hot.requires_grad = True
    
    # 计算目标类别的损失值
    # 将one-hot向量与模型输出相乘，然后求和
    class_vec = torch.sum(one_hot * ouput_vec)
    
    return class_vec


def gen_cam(feature_map, grads):
    """
    根据梯度和特征图生成CAM热力图
    
    这是Grad-CAM算法的核心函数，通过梯度加权特征图生成类激活映射
    
    @param feature_map: numpy.ndarray, 特征图，形状为[C, H, W]
    @param grads: numpy.ndarray, 梯度，形状为[C, H, W]
    @return: numpy.ndarray, CAM热力图，形状为[H, W]
    """
    # 初始化CAM数组，形状为特征图的空间维度(H, W)
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    
    # 计算每个通道的权重：对每个通道的梯度在空间维度上求平均
    # 这些权重表示每个通道对最终预测的重要程度
    weights = np.mean(grads, axis=(1, 2))
    
    # 将权重与对应的特征图相乘并累加
    # 这是Grad-CAM的核心公式：CAM = Σ(w_k * A_k)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    
    # 应用ReLU激活函数：只保留正值，负值设为0
    # 这确保了CAM只关注对预测有正面贡献的区域
    cam = np.maximum(cam, 0)
    
    # 将CAM调整到224x224尺寸，与输入图像保持一致
    cam = cv2.resize(cam, (224, 224))
    
    # 归一化CAM到[0,1]范围
    cam -= np.min(cam)  # 减去最小值，使最小值为0
    cam /= np.max(cam)  # 除以最大值，使最大值为1
    
    return cam


if __name__ == '__main__':
    # =============================== 主程序：Grad-CAM算法演示 ===============================
    
    # 设置文件路径和参数
    path_img = "both.png"                    # 输入图像路径
    path_cls_names = "imagenet1000.json"     # ImageNet类别名称文件
    output_dir = "./Result"                  # 结果输出目录
    input_size = 224                         # 模型输入尺寸

    # =============================== 模型和类别名称加载 ===============================
    # 加载ImageNet类别名称
    classes = load_class_names(path_cls_names)
    # 加载预训练的ResNet50模型
    resnet_50 = models.resnet50(pretrained=True)

    # =============================== Hook机制设置 ===============================
    # 创建全局列表来存储特征图和梯度信息
    fmap_block = []  # 存储前向传播的特征图
    grad_block = []  # 存储反向传播的梯度
    
    # 注册Hook函数到ResNet的最后一个卷积层
    # forward_hook: 在前向传播时捕获特征图
    # backward_hook: 在反向传播时捕获梯度
    resnet_50.layer4[-1].register_forward_hook(farward_hook)
    resnet_50.layer4[-1].register_full_backward_hook(backward_hook)

    # =============================== 图像预处理 ===============================
    # 读取输入图像
    img = cv2.imread(path_img, 1)  # H*W*C格式
    # 对图像进行预处理，转换为模型输入格式
    img_input = img_preprocess(img)

    # =============================== 前向传播 ===============================
    # 将预处理后的图像输入模型
    output = resnet_50(img_input)
    # 获取预测类别索引
    idx = np.argmax(output.cpu().data.numpy())
    # 打印预测结果
    print("predict: {}".format(classes[idx]))

    # =============================== 反向传播 ===============================
    # 清空之前的梯度
    resnet_50.zero_grad()
    # 计算目标类别的损失值
    class_loss = comp_class_vec(output)
    # 反向传播，计算梯度
    class_loss.backward()

    # =============================== CAM生成 ===============================
    # 从Hook中提取梯度和特征图
    grads_val = grad_block[0].cpu().data.numpy().squeeze()  # 梯度，形状[C, H, W]
    fmap = fmap_block[0].cpu().data.numpy().squeeze()       # 特征图，形状[C, H, W]
    
    # 使用梯度和特征图生成CAM热力图
    cam = gen_cam(fmap, grads_val)

    # =============================== 结果保存 ===============================
    # 将原始图像调整到指定尺寸并归一化到[0,1]范围
    img_show = np.float32(cv2.resize(img, (input_size, input_size))) / 255
    # 将CAM热力图叠加到原始图像上并保存
    show_cam_on_image(img_show, cam, output_dir)