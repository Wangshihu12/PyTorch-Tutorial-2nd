# -*- coding:utf-8 -*-
"""
@file name  : 02——optimizer.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-05-31
@brief      : 优化器基类属性与方法
@description: 演示PyTorch优化器的核心功能和使用方法
             包括参数组管理、梯度清零、状态字典查看和动态添加参数组等
             展示如何为不同参数设置不同的学习率和优化策略
"""
import torch
import torch.optim as optim


if __name__ == "__main__":
    # =================================== 参数组演示 ========================================
    # 参数组允许为不同的参数设置不同的优化策略（如学习率、动量等）
    
    # 创建三个需要梯度的权重张量
    w1 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w1.requires_grad = True  # 设置需要计算梯度

    w2 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w2.requires_grad = True  # 设置需要计算梯度

    w3 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w3.requires_grad = True  # 设置需要计算梯度

    # 演示1：一个参数组 - 所有参数使用相同的优化设置
    # 将w1和w3放在同一个参数组中，使用相同的学习率0.1
    optimizer_1 = optim.SGD([w1, w3], lr=0.1)
    print('len(optimizer.param_groups): ', len(optimizer_1.param_groups))  # 参数组数量：1
    print(optimizer_1.param_groups, '\n')  # 打印参数组详细信息

    # 演示2：两个参数组 - 不同参数使用不同的优化设置
    # w1使用学习率0.1，w2使用学习率0.001
    optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},      # 第一个参数组：w1，学习率0.1
                             {'params': w2, 'lr': 0.001}])    # 第二个参数组：w2，学习率0.001
    print('len(optimizer.param_groups): ', len(optimizer_2.param_groups))  # 参数组数量：2
    print(optimizer_2.param_groups)  # 打印参数组详细信息

    # =================================== zero_grad演示 ========================================
    # zero_grad()用于清零所有参数的梯度，防止梯度累积
    
    print("\n\n")
    
    # 重新创建权重张量
    w1 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w1.requires_grad = True  # 设置需要计算梯度

    w2 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w2.requires_grad = True  # 设置需要计算梯度

    # 创建SGD优化器，设置学习率和动量
    optimizer = optim.SGD([w1, w2], lr=0.001, momentum=0.9)

    # 手动设置w1的梯度（模拟反向传播后的梯度）
    # 通过参数组访问参数：optimizer.param_groups[0]['params'][0] 表示第一个参数组的第一个参数
    optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

    print('参数w1的梯度：')
    print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

    # 执行zero_grad()清零所有参数的梯度
    optimizer.zero_grad()
    print('执行zero_grad()之后，参数w1的梯度：')
    print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度
    
    # ------------------------- state_dict演示 -------------------------
    # state_dict()返回优化器的完整状态信息，包括参数组设置和优化器状态
    
    print("\n\n")
    print("state dict:{}".format(optimizer.state_dict()))  # 打印优化器的状态字典

    # =================================== add_param_group演示 ========================================
    # add_param_group()允许在训练过程中动态添加新的参数组
    
    print("\n\n")
    
    # 重新创建权重张量
    w1 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w1.requires_grad = True  # 设置需要计算梯度

    w2 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w2.requires_grad = True  # 设置需要计算梯度

    w3 = torch.randn(2, 2)  # 2x2的随机权重矩阵
    w3.requires_grad = True  # 设置需要计算梯度

    # 初始状态：一个参数组，包含w1和w2
    optimizer_1 = optim.SGD([w1, w2], lr=0.1)
    print('当前参数组个数: ', len(optimizer_1.param_groups))  # 参数组数量：2
    print(optimizer_1.param_groups, '\n')  # 打印参数组详细信息

    # 动态添加新的参数组：w3，使用不同的学习率和动量
    print('增加一组参数 w3\n')
    optimizer_1.add_param_group({'params': w3, 'lr': 0.001, 'momentum': 0.8})  # 添加w3参数组

    # 查看添加后的状态
    print('当前参数组个数: ', len(optimizer_1.param_groups))  # 参数组数量：3
    print(optimizer_1.param_groups, '\n')  # 打印更新后的参数组详细信息

