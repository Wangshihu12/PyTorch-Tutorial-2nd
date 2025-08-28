# -*- coding:utf-8 -*-
"""
@file name  : 02_make_grid.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-09
@brief      : make_grid 函数学习
"""
import os
import torch
import cv2
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


def resize_img_keep_ratio(img_, target_size):
    """
    按比例缩放图像，并填充至指定大小
    
    该函数保持图像的原始宽高比，将图像缩放到不超过目标尺寸的最大可能尺寸，
    然后在四周添加黑色填充，使最终图像达到指定的目标尺寸。
    
    @param img_: numpy.ndarray, 输入的图像数组，格式为HWC（高度、宽度、通道）
    @param target_size: tuple, 目标尺寸，格式为(高度, 宽度)
    @return: numpy.ndarray, 处理后的图像数组，尺寸为目标尺寸
    """
    # 获取原始图像的高度和宽度（前两个维度）
    old_size = img_.shape[0:2]  # 原始图像大小
    
    # 计算缩放比例：取宽高比例中的较小值，确保图像完全适应目标尺寸
    # 这样可以保持图像的原始宽高比，避免图像变形
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    
    # 根据计算出的比例，计算保持宽高比的新尺寸
    new_size = tuple([int(i * ratio) for i in old_size])
    
    # 使用OpenCV的resize函数调整图像尺寸
    # 注意：cv2.resize的参数顺序是(宽度, 高度)，所以需要交换new_size的顺序
    img = cv2.resize(img_, (new_size[1], new_size[0]))
    
    # 计算需要在宽度和高度方向上填充的像素数量
    pad_w = target_size[1] - new_size[1]  # 宽度方向需要填充的像素数
    pad_h = target_size[0] - new_size[0]  # 高度方向需要填充的像素数
    
    # 计算上下左右的填充像素数，确保图像居中
    # 使用整除和取余操作，处理奇数像素的情况
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)      # 上边和下边的填充
    left, right = pad_w // 2, pad_w - (pad_w // 2)      # 左边和右边的填充
    
    # 使用OpenCV的copyMakeBorder函数添加边框填充
    # cv2.BORDER_CONSTANT: 使用常数填充
    # (0, 0, 0): 填充颜色为黑色（BGR格式）
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    
    return img_new


if __name__ == "__main__":
    # 创建TensorBoard写入器，用于记录图像网格的可视化结果
    writer = SummaryWriter(comment="grid images", filename_suffix="grid_img")
    
    # ============================== 模式1：真实图像网格可视化 ==============================
    # 数据集下载链接
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    
    # 设置数据目录路径（COVID-19数据集的图像文件夹）
    data_dir = r"F:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\imgs"  # 你的数据路径
    
    # 获取目录下所有图像文件的名称
    name_list = os.listdir(data_dir)
    # 构建完整的图像文件路径列表
    path_list = [os.path.join(data_dir, name) for name in name_list]

    # 设置目标图像尺寸为500x500像素
    PATCH_SIZE = (500, 500)
    
    # 创建图像列表，用于存储处理后的图像张量
    img_list = []
    
    # 遍历每个图像文件路径
    for path_img in path_list:
        # 使用OpenCV读取图像（BGR格式）
        img_hwc = cv2.imread(path_img)
        
        # 调用自定义函数调整图像尺寸，保持宽高比并填充至目标尺寸
        img_resize = resize_img_keep_ratio(img_hwc, PATCH_SIZE)
        
        # 将numpy数组转换为PyTorch张量
        img_tensor = torch.from_numpy(img_resize)
        
        # 调整张量维度顺序：HWC --> CHW
        # 第一次transpose(0, 2): HWC --> CWH
        # 第二次transpose(1, 2): CWH --> CHW
        img_tensor = img_tensor.transpose(0, 2).transpose(1, 2)
        
        # 将处理后的图像张量添加到列表中
        img_list.append(img_tensor)
    
    # 使用make_grid函数将所有图像排列成网格
    # normalize=False: 不进行归一化，保持原始像素值
    # scale_each=False: 不对每个图像单独缩放
    img_grid = vutils.make_grid(img_list, normalize=False, scale_each=False)
    
    # 将图像网格添加到TensorBoard中
    writer.add_image("X-ray", img_grid)

    # ============================== 模式2：随机图像网格可视化 ==============================
    # 循环生成10组随机图像网格，演示动态可视化效果
    for step in range(10):
        # 生成随机图像张量：32个批次，3个通道，64x64像素
        # 格式：(B x C x H x W) = (32, 3, 64, 64)
        dummy_img = torch.rand(32, 3, 64, 64)
        
        # 使用make_grid函数将随机图像排列成网格
        # normalize=True: 将值归一化到[0,1]范围
        # scale_each=True: 每个图像单独归一化，便于观察
        img_grid = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        
        # 将图像网格添加到TensorBoard中，并记录步数
        writer.add_image('Image', img_grid, step)

    # 关闭TensorBoard写入器，确保所有数据都被写入磁盘
    writer.close()















