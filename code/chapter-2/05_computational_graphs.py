# -*- coding:utf-8 -*-
"""
@file name  : 05_computational_graphs.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-06
@brief      : 计算图中的叶子结点观察
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd.function import Function

if __name__ == "__main__":
    import torch

    # 创建两个需要梯度的叶子张量（叶子节点）
    # requires_grad=True 表示这些张量需要计算梯度
    w = torch.tensor([1.], requires_grad=True)  # 叶子节点1：权重参数
    x = torch.tensor([2.], requires_grad=True)  # 叶子节点2：输入数据

    # 构建计算图：执行前向传播操作
    # 每个操作都会创建一个新的张量，并记录其计算历史
    a = torch.add(w, x)      # 计算 a = w + x = 1 + 2 = 3
    b = torch.add(w, 1)      # 计算 b = w + 1 = 1 + 1 = 2
    y = torch.mul(a, b)      # 计算 y = a * b = 3 * 2 = 6

    # 反向传播：计算梯度
    # 从输出 y 开始，沿着计算图反向传播梯度
    y.backward()
    
    # 打印 w 的梯度
    # 根据链式法则：∂y/∂w = ∂y/∂a * ∂a/∂w + ∂y/∂b * ∂b/∂w
    # ∂y/∂a = b = 2, ∂a/∂w = 1, ∂y/∂b = a = 3, ∂b/∂w = 1
    # 所以 ∂y/∂w = 2*1 + 3*1 = 5
    print(w.grad)

    # 查看各个张量的叶子节点属性
    # is_leaf=True 表示该张量是用户直接创建的，不是通过计算得到的
    # is_leaf=False 表示该张量是通过计算操作得到的中间结果
    print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
    
    # 查看各个张量的梯度
    # 只有叶子节点（w, x）的梯度会被保留
    # 中间节点（a, b, y）的梯度在反向传播后会被释放以节省内存
    print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
    
    # 查看各个张量的 grad_fn 属性
    # grad_fn 记录了创建该张量时使用的操作（函数）
    # None 表示该张量是叶子节点，没有 grad_fn
    # 非None 表示该张量是通过某个操作创建的，grad_fn 指向该操作
    print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
