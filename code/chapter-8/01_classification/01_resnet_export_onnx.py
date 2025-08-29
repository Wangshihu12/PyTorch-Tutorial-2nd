# -*- coding:utf-8 -*-
"""
@file name  : 01_resnet_export_onnx.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-02
@brief      : resnet50 onnx导出
"""
import os.path

import torchvision
import torch
import torch
import torchvision
import torch.nn as nn

# 定义预训练模型权重文件的路径
ckpt_path = r"./Result/2023-09-25_22-09-35/checkpoint_best.pth"
# 设置设备：如果CUDA可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建ResNet50模型实例，不加载预训练权重
model = torchvision.models.resnet50(pretrained=False)

# 替换第一层卷积层：因为预训练模型输入是3通道RGB图像，而本案例是灰度图，输入是1通道
# 参数说明：输入通道数=1（灰度图），输出通道数=64，卷积核大小=7x7，步长=2x2，填充=3x3，不使用偏置项
model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# 获取全连接层的输入特征维度
num_ftrs = model.fc.in_features  # 替换最后一层
# 替换最后的全连接层：输入特征维度保持不变，输出维度改为2（二分类任务）
model.fc = nn.Linear(num_ftrs, 2)

# 加载预训练模型权重文件
state_dict = torch.load(ckpt_path)
# 从权重字典中提取模型状态字典
model_sate_dict = state_dict['model_state_dict']
# 将预训练权重加载到模型中
model.load_state_dict(model_sate_dict)  # 模型参数加载


if __name__ == '__main__':

    # 设置ONNX操作集版本为13
    op_set = 13
    # 创建虚拟输入数据：批次大小=1，通道数=3，图像尺寸=224x224
    # 注意：这里通道数仍然是3，与之前修改的1通道不一致，可能是代码遗留问题
    dummy_data = torch.randn((1, 3, 224, 224))

    # 固定批次大小为1，设置输出目录和文件名
    out_dir = os.path.dirname(ckpt_path)  # 获取权重文件所在目录
    path_out = os.path.join(out_dir, "resnet50_bs_1.onnx")  # 构建ONNX文件输出路径
    
    # 导出模型为ONNX格式
    # 参数说明：
    # model: 要导出的PyTorch模型
    # (dummy_data): 虚拟输入数据，用于确定输入形状
    # path_out: ONNX文件输出路径
    # opset_version: ONNX操作集版本
    # input_names: 输入张量的名称
    # output_names: 输出张量的名称
    torch.onnx.export(model, (dummy_data), path_out,
                      opset_version=op_set, input_names=['input'],  output_names=['output'])



