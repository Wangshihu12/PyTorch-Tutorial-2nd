# -*- coding:utf-8 -*-
"""
@file name  : 05_model_print.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-21
@brief      : 模型参数打印
"""
import torchvision.models as models
from torchinfo import summary

if __name__ == '__main__':
    # =============================== 模型创建和配置 ===============================
    # 创建ResNet50模型实例
    # pretrained=False: 使用随机初始化的权重，不加载预训练权重
    # 这样可以避免下载预训练模型，加快演示速度
    resnet_50 = models.resnet50(pretrained=False)
    
    # 设置批次大小为1，用于模型信息统计
    # 批次大小影响内存使用和计算复杂度
    batch_size = 1

    # =============================== 基本模型信息统计 ===============================
    # 使用torchinfo的summary函数打印模型的基本信息
    # input_size: 指定输入张量的形状 (batch_size, channels, height, width)
    # 对于ResNet50，标准输入尺寸是224x224像素，3个颜色通道
    summary(resnet_50, input_size=(batch_size, 3, 224, 224))

    # =============================== 自定义列名配置 ===============================
    # 定义自定义的列名，用于控制summary输出的列显示
    # 这些列名对应模型统计信息的不同方面
    col_names_ = (
        "input_size",      # 输入尺寸：显示每层的输入张量形状
        "output_size",     # 输出尺寸：显示每层的输出张量形状
        "num_params",      # 参数数量：显示每层的可训练参数总数
        "kernel_size",     # 卷积核尺寸：显示卷积层的核大小
        "mult_adds",       # 乘加运算：显示每层的计算复杂度
        "trainable",       # 是否可训练：显示参数是否参与梯度更新
    )
    
    # 使用自定义列名显示模型信息（当前被注释）
    # 这样可以只显示我们关心的特定信息列
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), col_names=col_names_)

    # =============================== 其他配置选项示例 ===============================
    # 选项1：只显示输入尺寸列
    # 适用于只需要了解模型各层输入输出形状的场景
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), col_names=("input_size",))
    
    # 选项2：设置行显示格式为ASCII字符
    # 适用于在某些终端环境下需要兼容ASCII字符显示的情况
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), row_settings=("ascii_only",))
    
    # 选项3：启用详细输出模式
    # verbose=1: 显示更详细的模型信息，包括每层的具体参数
    # 适用于需要深入了解模型结构的场景
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), verbose=1)
