# -*- coding:utf-8 -*-
"""
@file name  : 05_torchmetrics.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : torchmetrics库学习 - 演示如何使用官方torchmetrics库进行准确率计算和混淆矩阵分析
             
             参考项目：https://github.com/Lightning-AI/metrics
             
             torchmetrics是PyTorch生态中标准的评估指标库，提供了丰富的预定义指标
             和统一的接口，支持分布式训练和自动状态管理
"""


if __name__ == "__main__":
    # =============================== 主程序：演示torchmetrics.Accuracy的使用 ===============================
    
    # 导入工具函数
    from my_utils import setup_seed
    
    # 设置随机种子，确保实验可重复性
    # 这对于调试和结果复现非常重要
    setup_seed(40)
    
    # 导入必要的库
    import torch
    import torchmetrics

    # 创建torchmetrics的Accuracy指标实例
    # torchmetrics.Accuracy是官方提供的准确率计算器
    # 内部维护了tp(真正例)、tn(真负例)、fp(假正例)、fn(假负例)四个状态变量
    metric = torchmetrics.Accuracy()
    
    # 设置批次数，模拟多批次训练过程
    n_batches = 3
    
    # 模拟多个批次的训练过程
    for i in range(n_batches):
        # 生成随机预测结果
        # torch.randn(10, 5): 生成10个样本，每个样本5个类别的随机值
        # 这些值通常来自神经网络的最后一层（logits）
        preds = torch.randn(10, 5).softmax(dim=-1)
        
        # 生成随机真实标签
        # torch.randint(5, (10,)): 生成10个0-4之间的随机整数作为标签
        # 这模拟了真实数据集中的标签分布
        target = torch.randint(5, (10,))
        
        # 计算当前批次的准确率
        # metric(preds, target) 等价于：
        # 1. metric.update(preds, target) - 更新内部状态
        # 2. metric.compute() - 计算当前累积的准确率
        # 
        # 注意：这里每次调用都会计算累积到当前批次的准确率
        # 通过维护tp, tn, fp, fn四个状态变量来记录所有批次的数据
        acc = metric(preds, target)
        
        # 打印当前批次的准确率
        print(f"Accuracy on batch {i}: {acc}")

    # =============================== 全局准确率计算 ===============================
    # 计算所有数据的全局准确率
    # 基于累积的tp, tn, fp, fn状态变量计算
    acc_avg = metric.compute()
    print(f"Accuracy on all data: {acc_avg}")
    
    # =============================== 混淆矩阵状态分析 ===============================
    # 提取内部维护的混淆矩阵状态变量
    # 这些变量记录了所有批次的累积统计信息
    tp = metric.tp  # 真正例：预测为正类且实际为正类的样本数
    tn = metric.tn  # 真负例：预测为负类且实际为负类的样本数
    fp = metric.fp  # 假正例：预测为正类但实际为负类的样本数
    fn = metric.fn  # 假负例：预测为负类但实际为正类的样本数
    
    # 打印混淆矩阵的各个组成部分
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    
    # 验证状态变量的总和
    # 总和应该等于所有批次的样本总数：n_batches * 10 = 30
    total_samples = sum([tp, tn, fp, fn])
    print(f"Total samples: {total_samples}")
    
    # =============================== 状态重置 ===============================
    # 重置指标状态，准备下一轮计算
    # 这会清空所有累积的tp, tn, fp, fn状态
    metric.reset()
    
    # 验证重置后的状态
    print(f"After reset - TP: {metric.tp}, TN: {metric.tn}, FP: {metric.fp}, FN: {metric.fn}")




