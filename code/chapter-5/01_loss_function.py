# -*- coding:utf-8 -*-
"""
@file name  : 01_loss_function.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-05-29
@brief      : loss function实现流程剖析
@description: 演示PyTorch中常用损失函数的使用方法和参数设置
             包括L1损失函数和交叉熵损失函数的各种reduction模式
             展示权重设置和忽略索引等高级功能
"""
import torch
import torch.nn as nn


if __name__ == "__main__":
    # ========================== L1损失函数演示 ==========================
    # L1损失函数（Mean Absolute Error, MAE）：计算预测值与真实值的绝对差值的平均
    
    # 创建模拟数据：输出张量和目标张量
    output = torch.ones(2, 2, requires_grad=True) * 0.5  # 2x2的输出张量，所有值为0.5，需要梯度
    target = torch.ones(2, 2)  # 2x2的目标张量，所有值为1.0

    # 定义三种reduction模式：none（不减少）、mean（求平均）、sum（求和）
    params = "none mean sum".split()

    # 遍历不同的reduction模式，观察损失值的差异
    for p in params:
        loss_func = nn.L1Loss(reduction=p)  # 创建L1损失函数实例
        loss_tmp = loss_func(output, target)  # 计算损失值
        print("reduction={}:loss={}, shape:{}".format(p, loss_tmp, loss_tmp.shape))

    print("\n")
    
    # ========================== 交叉熵损失函数演示 ==========================
    # 交叉熵损失函数：用于多分类问题，结合了Softmax和NLLLoss
    
    # 重新导入必要的库（避免重复导入）
    import torch
    import torch.nn as nn
    import numpy as np
    import math

    # 定义三种reduction模式
    params = "none mean sum".split()

    # 创建模拟数据：三分类任务，批次大小为2
    output = torch.ones(2, 3, requires_grad=True) * 0.5  # 2个样本，每个样本3个类别的预测值
    target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)  # 真实标签：第0个样本属于类别0，第1个样本属于类别1

    # 遍历不同的reduction模式，观察损失值的差异
    for p in params:
        loss_func = nn.CrossEntropyLoss(reduction=p)  # 创建交叉熵损失函数实例
        loss_tmp = loss_func(output, target)  # 计算损失值
        print("reduction={}:loss={}, shape:{}".format(p, loss_tmp, loss_tmp.shape))

    # ----------------------- 手动计算交叉熵损失 ------------------------------------
    # 熟悉计算公式，手动计算第一个样本的损失值
    # 交叉熵损失公式：Loss = -log(exp(x_class) / sum(exp(x_i)))
    output = output[0].detach().numpy()  # 获取第一个样本的输出，转换为numpy数组
    output_1 = output[0]  # 第一个样本的输出值
    target_1 = target[0].numpy()  # 第一个样本的真实标签
    
    # 第一项：-x_class（负的对数概率）
    x_class = output[target_1]  # 获取真实类别对应的输出值
    
    # 第二项：log(sum(exp(x_i)))（对数归一化项）
    exp = math.e  # 自然常数e
    sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2])  # 计算指数和
    log_sigma_exp_x = math.log(sigma_exp_x)  # 取对数
    
    # 两项相加得到最终的损失值
    loss_1 = -x_class + log_sigma_exp_x
    print("\n手动计算，第一个样本的loss:{}".format(loss_1))

    # ----------------------- 权重设置演示 ------------------------------------
    # 为不同类别设置不同的权重，用于处理类别不平衡问题
    
    # 定义类别权重：类别0权重0.6，类别1权重0.2，类别2权重0.2
    weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
    loss_f = nn.CrossEntropyLoss(weight=weight, reduction="none")  # 创建带权重的损失函数
    
    # 重新创建模拟数据
    output = torch.ones(2, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
    target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)  # 真实标签
    
    loss = loss_f(output, target)  # 计算带权重的损失值
    print('\n\nCrossEntropy loss: weight')
    print('loss: ', loss)  # 打印损失值
    print('原始loss值为1.0986, 第一个样本是第0类，weight=0.6,所以输出为1.0986*0.6 =', 1.0986 * 0.6)

    # ----------------------- 忽略索引演示 ------------------------------------
    # ignore_index参数：忽略指定类别的损失计算，常用于处理填充标签或特殊类别
    
    # 创建两个不同的损失函数，分别忽略不同的类别
    loss_f_1 = nn.CrossEntropyLoss(weight=None, reduction="none", ignore_index=1)  # 忽略类别1
    loss_f_2 = nn.CrossEntropyLoss(weight=None, reduction="none", ignore_index=2)  # 忽略类别2

    # 创建模拟数据：三分类任务，批次大小为3
    output = torch.ones(3, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为3个，假设每个神经元输出都为0.5
    target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)  # 真实标签：0, 1, 2

    # 计算两种不同ignore_index设置的损失值
    loss_1 = loss_f_1(output, target)  # 忽略类别1的损失
    loss_2 = loss_f_2(output, target)  # 忽略类别2的损失

    print('\n\nCrossEntropy loss: ignore_index')
    print('\nignore_index = 1: ', loss_1)  # 类别为1的样本的loss为0（被忽略）
    print('ignore_index = 2: ', loss_2)  # 类别为2的样本的loss为0（被忽略）

