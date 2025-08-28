# -*- coding:utf-8 -*-
"""
@file name  : 01_tensorboard.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-06
@brief      : tensorboard 安装与使用
"""
import torch
import torch.nn as nn


if __name__ == "__main__":
    # 导入操作系统相关模块
    import os

    # 获取当前文件的绝对路径，然后获取其所在目录
    # 这样设置可以确保日志文件保存在脚本所在的目录中
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 从PyTorch导入TensorBoard的SummaryWriter类
    # SummaryWriter是PyTorch中用于记录训练日志的主要接口
    from torch.utils.tensorboard import SummaryWriter

    # 设置日志目录为当前脚本所在目录
    log_dir = BASE_DIR  # 即test_tensorboard.py文件所在目录
    
    # 创建SummaryWriter实例，用于记录各种训练指标和可视化数据
    # log_dir: 指定日志文件的保存目录
    # filename_suffix: 为日志文件名添加后缀，便于识别和管理
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_test_tensorboard")
    
    # 另一种创建方式（当前被注释）：
    # comment参数会在日志目录名后添加注释，便于区分不同的实验
    # writer = SummaryWriter(comment="test01", filename_suffix="_test_tensorboard")
    
    # 创建x轴数据：0到99的整数序列，共100个点
    x = range(100)
    
    # 循环记录两种不同的标量数据到TensorBoard
    for i in x:
        # 记录线性函数 y = 2x 的值
        # 第一个参数：标签名称，用于在TensorBoard中显示
        # 第二个参数：标量值，即 y = 2x 的计算结果
        # 第三个参数：全局步数，通常对应训练轮数或迭代次数
        writer.add_scalar('y=2x', i * 2, i)
        
        # 记录指数函数 y = 2^x 的值
        # 这个函数会快速增长，展示TensorBoard处理不同数值范围的能力
        writer.add_scalar('y=pow(2, x)', 2 ** i, i)
    
    # 关闭SummaryWriter，确保所有数据都被写入磁盘
    # 这是一个重要的步骤，避免数据丢失
    writer.close()
