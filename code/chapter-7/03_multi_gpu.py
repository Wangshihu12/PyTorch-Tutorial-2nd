# -*- coding: utf-8 -*-
"""
# @file name  : 03_multi_gpu.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : 多gpu示例。单机多卡情况。演示如何使用DataParallel实现多GPU并行训练
"""
import torch
import torch.nn as nn

# =============================== 设备配置 ===============================
# 自动检测并使用可用的设备（GPU或CPU）
# 如果CUDA可用，则使用GPU；否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FooNet(nn.Module):
    """
    示例神经网络类，用于演示多GPU并行训练
    
    这是一个简单的全连接网络，包含多个线性层和ReLU激活函数
    主要用于演示DataParallel的工作原理
    """
    def __init__(self, neural_num, layers=3):
        """
        初始化网络结构
        
        @param neural_num: int, 每层神经元的数量，决定了输入输出维度
        @param layers: int, 网络层数，默认为3层
        """
        super(FooNet, self).__init__()
        
        # 创建ModuleList，包含指定数量的线性层
        # 每层都是neural_num x neural_num的线性变换，不使用偏置项
        # 这样设计使得每层的输入输出维度保持一致
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):
        """
        前向传播函数
        
        @param x: torch.Tensor, 输入张量，形状为(batch_size, neural_num)
        @return: torch.Tensor, 经过网络处理后的输出张量，形状与输入相同
        """
        # 打印当前批次大小，用于调试和监控
        # 在多GPU环境下，这个值可能因数据分发而有所不同
        print("\nbatch size in forward: {}".format(x.size()[0]))

        # 依次通过每一层线性层和ReLU激活函数
        for (i, linear) in enumerate(self.linears):
            x = linear(x)        # 线性变换：y = Wx
            x = torch.relu(x)    # ReLU激活函数：max(0, x)，增加非线性
            
        return x


if __name__ == "__main__":
    # =============================== 主程序入口 ===============================
    
    # 设置批次大小
    batch_size = 16  # 总批次大小，会被分配到多个GPU上

    # =============================== 数据准备 ===============================
    # 生成随机输入数据
    # inputs: 形状为(16, 3)的张量，表示16个样本，每个样本3个特征
    inputs = torch.randn(batch_size, 3)
    
    # 生成随机标签数据（虽然在这个示例中未使用）
    # labels: 形状为(16, 3)的张量，与输入维度匹配
    labels = torch.randn(batch_size, 3)

    # 将数据移动到指定设备（GPU或CPU）
    # 这是多GPU训练的必要步骤，确保数据在正确的设备上
    inputs, labels = inputs.to(device), labels.to(device)

    # =============================== 模型构建 ===============================
    # 创建网络实例
    net = FooNet(neural_num=3, layers=3)
    
    # 使用DataParallel包装网络，实现多GPU并行训练
    # DataParallel会自动：
    # 1. 将输入数据分发到多个GPU上
    # 2. 在每个GPU上并行计算前向传播
    # 3. 将多个GPU的结果合并
    # 4. 处理反向传播的梯度同步
    net = nn.DataParallel(net)
    
    # 将包装后的网络移动到指定设备
    # 注意：DataParallel会自动处理多GPU间的数据分发
    net.to(device)

    # =============================== 训练过程 ===============================
    # 模拟训练循环（这里只执行1个epoch作为演示）
    for epoch in range(1):
        # 前向传播
        # 在DataParallel环境下，输入数据会自动分发到多个GPU
        # 每个GPU处理batch_size/GPU数量的样本
        outputs = net(inputs)

        # 打印模型输出的尺寸
        # 输出尺寸应该与输入尺寸相同：(batch_size, neural_num)
        print("model outputs.size: {}".format(outputs.size()))

    # =============================== 设备信息 ===============================
    # 打印可用的CUDA设备数量
    # 这个值表示系统中有多少个可用的GPU
    print("device_count :{}".format(torch.cuda.device_count()))



