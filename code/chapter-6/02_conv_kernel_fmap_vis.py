# -*- coding:utf-8 -*-
"""
@file name  : 02_conv_kernel_fmap_vis.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-09
@brief      : 卷积核可视化，特征图可视化
"""
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models


if __name__ == "__main__":
    # =============================== 卷积核可视化部分 ===============================
    # 创建TensorBoard写入器，用于记录卷积核可视化结果
    writer = SummaryWriter(comment='kernel', filename_suffix="_test_your_filename_suffix")
    
    # 加载预训练的AlexNet模型
    alexnet = models.alexnet(pretrained=True)
    
    # 初始化卷积层计数器（从-1开始，因为第一次遇到卷积层时会+1变成0）
    kernel_num = -1
    # 设置最大可视化层数，只可视化前vis_max层卷积层
    vis_max = 1
    
    # 遍历AlexNet模型中的所有模块（层）
    for sub_module in alexnet.modules():
        # 只处理卷积层，非卷积层则跳过
        if isinstance(sub_module, nn.Conv2d):
            # 遇到卷积层，计数器加1
            kernel_num += 1
            # 如果超过预期可视化的层数，则停止遍历
            if kernel_num > vis_max:
                break

            # 获取当前卷积层的权重参数，即卷积核权重
            kernels = sub_module.weight
            # 解析卷积核的维度：输出通道数、输入通道数、卷积核宽度、卷积核高度
            c_out, c_int, k_w, k_h = tuple(kernels.shape)

            # 遍历每个输出通道的卷积核
            for o_idx in range(c_out):
                # 提取第o_idx个输出通道的所有卷积核
                # 一个卷积核是4D的，包括输入通道和输出通道
                # 这里将每一个二维矩阵看作一个最细粒度的卷积核，进行绘制
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # make_grid需要BCHW格式，这里拓展C维度
                
                # 使用make_grid将卷积核排列成网格形式，便于可视化
                # normalize=True: 将值归一化到[0,1]范围
                # scale_each=True: 每个卷积核单独归一化
                # nrow=c_int: 每行显示c_int个卷积核
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                
                # 将卷积核网格图像添加到TensorBoard中
                # 标签名包含层数和通道信息
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

            # 对当前层的所有卷积核进行整体可视化
            # 将卷积核重新排列为(-1, 3, k_h, k_w)的格式，便于显示
            kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3表示RGB三个通道
            # 创建网格，每行显示8个卷积核
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
            
            # 将整体卷积核网格添加到TensorBoard中
            writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=42)

            # 打印当前卷积层的形状信息，便于调试
            print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    # 关闭卷积核可视化的TensorBoard写入器
    writer.close()
    
    # =============================== 特征图可视化部分 ===============================
    # 创建新的TensorBoard写入器，用于记录特征图可视化结果
    writer = SummaryWriter(comment='fmap_vis', filename_suffix="_test_your_filename_suffix")

    # =============================== 数据预处理 ===============================
    # 图像路径设置
    # 可以使用任何图像，这里使用经典的Lena图像
    # 提示：Lena(Lena Soderberg, 莱娜·瑟德贝里)是图像处理领域的标准测试图像
    path_img = r"F:\pytorch-tutorial-2nd\data\imgs\lena.png"  # 你的图像路径
    
    # 定义图像标准化参数（ImageNet数据集的标准化参数）
    normMean = [0.49139968, 0.48215827, 0.44653124]  # RGB三个通道的均值
    normStd = [0.24703233, 0.24348505, 0.26158768]   # RGB三个通道的标准差
    
    # 创建标准化变换
    norm_transform = transforms.Normalize(normMean, normStd)
    
    # 组合图像预处理变换
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),    # 调整图像尺寸为224x224（AlexNet的输入尺寸）
        transforms.ToTensor(),            # 转换为PyTorch张量
        norm_transform                    # 应用标准化
    ])

    # 加载并预处理图像
    img_pil = Image.open(path_img).convert('RGB')  # 打开图像并转换为RGB格式
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)        # 应用预处理变换
    img_tensor.unsqueeze_(0)  # 添加batch维度：chw --> bchw

    # =============================== 模型前向传播 ===============================
    # 重新加载AlexNet模型（因为之前已经关闭了writer）
    alexnet = models.alexnet(pretrained=True)

    # 获取第一层卷积层
    convlayer1 = alexnet.features[0]
    # 将预处理后的图像输入到第一层卷积层，得到特征图
    fmap_1 = convlayer1(img_tensor)

    # =============================== 特征图可视化 ===============================
    # 预处理特征图，调整维度顺序以便可视化
    # 原始格式：bchw=(1, 64, 55, 55)
    # 调整后格式：(64, 1, 55, 55)，将64个通道作为独立的图像进行显示
    fmap_1.transpose_(0, 1)
    
    # 使用make_grid将特征图排列成网格形式
    # 每行显示8个特征图，便于观察不同通道的激活模式
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    # 将特征图网格添加到TensorBoard中
    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    
    # 关闭特征图可视化的TensorBoard写入器
    writer.close()














