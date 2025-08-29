# -*- coding: utf-8 -*-
"""
# @file name  : 03_model_load_in_gpu.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : DataParallel的保存与加载 - 演示单GPU和多GPU环境下模型的保存与加载方法
"""

import os
import torch
import torch.nn as nn


class FooNet(nn.Module):
    """
    示例神经网络类，用于演示GPU环境下的模型保存与加载
    
    这是一个简单的全连接网络，包含多个线性层和ReLU激活函数
    """
    def __init__(self, neural_num, layers=3):
        """
        初始化网络结构
        
        @param neural_num: int, 每层神经元的数量
        @param layers: int, 网络层数，默认为3
        """
        super(FooNet, self).__init__()
        # 创建ModuleList，包含指定数量的线性层
        # 每层都是neural_num x neural_num的线性变换，不使用偏置项
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):
        """
        前向传播函数
        
        @param x: torch.Tensor, 输入张量，形状为(batch_size, neural_num)
        @return: torch.Tensor, 经过网络处理后的输出张量
        """
        # 打印当前批次大小，用于调试和监控
        print("\nbatch size in forward: {}".format(x.size()[0]))

        # 依次通过每一层线性层和ReLU激活函数
        for (i, linear) in enumerate(self.linears):
            x = linear(x)        # 线性变换
            x = torch.relu(x)    # ReLU激活函数，增加非线性
        return x


# =============================== 场景1：单GPU环境下的模型保存与加载 ===============================
flag = 0  # 控制是否执行此段代码
# flag = 1  # 设置为1时执行
if flag:
    # 设置要使用的GPU设备ID
    gpu_list = [0]  # 只使用第0号GPU
    
    # 将GPU列表转换为环境变量格式的字符串
    gpu_list_str = ','.join(map(str, gpu_list))
    
    # 设置CUDA环境变量，限制可见的GPU设备
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    
    # 检测可用的设备并创建设备对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建网络实例
    net = FooNet(neural_num=3, layers=3)
    
    # 将网络移动到指定设备（GPU或CPU）
    net.to(device)

    # =============================== 保存模型 ===============================
    # 获取网络的参数字典（state_dict）
    net_state_dict = net.state_dict()
    
    # 设置保存路径
    path_state_dict = "./model_in_gpu_0.pkl"
    
    # 将模型参数保存到文件
    torch.save(net_state_dict, path_state_dict)

    # =============================== 加载模型 ===============================
    # 方法1：直接加载（可能出错，因为保存时在GPU上）
    # state_dict_load = torch.load(path_state_dict)
    
    # 方法2：指定加载到CPU（推荐做法）
    # map_location="cpu" 确保模型参数被加载到CPU内存中
    state_dict_load = torch.load(path_state_dict, map_location="cpu")
    
    # 打印加载的参数信息
    print("state_dict_load:\n{}".format(state_dict_load))


# =============================== 场景2：多GPU环境下的模型保存 ===============================
flag = 0  # 控制是否执行此段代码
# flag = 1  # 设置为1时执行
if flag:
    # 检查可用的GPU数量
    if torch.cuda.device_count() < 2:
        print("gpu数量不足，请到多gpu环境下运行")
        import sys
        sys.exit(0)  # 如果GPU数量不足，直接退出程序

    # 设置要使用的GPU设备ID列表
    gpu_list = [0, 1, 2, 3]  # 使用4个GPU
    
    # 将GPU列表转换为环境变量格式的字符串
    gpu_list_str = ','.join(map(str, gpu_list))
    
    # 设置CUDA环境变量，限制可见的GPU设备
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    
    # 检测可用的设备并创建设备对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建网络实例
    net = FooNet(neural_num=3, layers=3)
    
    # 使用DataParallel包装网络，实现多GPU并行训练
    # DataParallel会自动将数据分发到多个GPU上，并合并结果
    net = nn.DataParallel(net)
    
    # 将网络移动到指定设备
    net.to(device)

    # =============================== 保存多GPU模型 ===============================
    # 获取网络的参数字典（包含DataParallel的包装信息）
    net_state_dict = net.state_dict()
    
    # 设置保存路径
    path_state_dict = "./model_in_multi_gpu.pkl"
    
    # 将模型参数保存到文件
    torch.save(net_state_dict, path_state_dict)


# =============================== 场景3：多GPU环境下的模型加载 ===============================
# flag = 0  # 控制是否执行此段代码
flag = 1  # 设置为1时执行
if flag:
    # 创建新的网络实例（注意：这里没有使用DataParallel包装）
    net = FooNet(neural_num=3, layers=3)

    # 设置要加载的模型文件路径
    path_state_dict = "./model_in_multi_gpu.pkl"
    
    # 加载模型参数到CPU
    state_dict_load = torch.load(path_state_dict, map_location="cpu")
    
    # 打印加载的参数信息
    print("state_dict_load:\n{}".format(state_dict_load))

    # =============================== 直接加载会出错 ===============================
    # 直接加载会失败，因为保存的模型包含DataParallel的"module."前缀
    # net.load_state_dict(state_dict_load)

    # =============================== 处理DataParallel前缀 ===============================
    # 创建新的有序字典来存储处理后的参数
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    # 遍历加载的参数，移除"module."前缀
    for k, v in state_dict_load.items():
        # 如果参数名以"module."开头，则去掉前7个字符
        # 这是因为DataParallel会自动为参数添加"module."前缀
        namekey = k[7:] if k.startswith('module.') else k
        
        # 将处理后的参数名和值添加到新字典中
        new_state_dict[namekey] = v
    
    # 打印处理后的参数字典
    print("new_state_dict:\n{}".format(new_state_dict))

    # 使用处理后的参数加载到网络中
    net.load_state_dict(new_state_dict)




















