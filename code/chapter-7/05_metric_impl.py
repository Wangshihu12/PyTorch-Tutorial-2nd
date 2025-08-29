# -*- coding:utf-8 -*-
"""
@file name  : 05_metric_impl.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : 自定义 metric - 演示如何使用torchmetrics框架实现自定义评估指标
"""

import torch
from torchmetrics import Metric


class MyAccuracy(Metric):
    """
    自定义准确率评估指标类
    
    继承自torchmetrics.Metric，实现了一个标准的准确率计算器
    支持批次更新和全局统计，适用于多批次数据的准确率计算
    
    特点：
    1. 支持分布式训练（通过dist_reduce_fx="sum"）
    2. 自动状态管理（correct和total计数）
    3. 支持批次累积和全局计算
    """
    
    # 类属性：控制状态更新策略
    # False表示每次update后不立即计算，只在调用compute时计算
    # 这样可以累积多个批次的数据，然后一次性计算全局准确率
    full_state_update: bool = False

    def __init__(self):
        """
        初始化准确率指标
        
        设置内部状态变量，用于累积正确预测数和总样本数
        """
        super().__init__()  # 调用父类Metric的初始化方法
        
        # 添加状态变量"correct"，用于累积正确预测的数量
        # default: 初始值为0的张量
        # dist_reduce_fx="sum": 在分布式训练中，多个进程的结果通过求和合并
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        
        # 添加状态变量"total"，用于累积总样本数量
        # 同样支持分布式训练的结果合并
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        更新指标状态
        
        处理一个批次的预测结果和真实标签，更新内部统计信息
        
        @param preds: torch.Tensor, 模型预测结果，形状为(batch_size, num_classes)
                      通常是经过softmax后的概率分布
        @param target: torch.Tensor, 真实标签，形状为(batch_size,)
                      包含每个样本的真实类别索引
        """
        # 获取当前批次的样本数量
        batch_size = target.size(0)
        
        # 获取Top1预测结果
        # preds.topk(1, 1, True, True) 返回：
        # - 第一个返回值：最大概率值（这里用_忽略）
        # - 第二个返回值：最大概率对应的索引（即预测的类别）
        # 参数说明：k=1（取前1个），dim=1（在类别维度上），largest=True（取最大值），sorted=True（排序）
        _, pred = preds.topk(1, 1, True, True)
        
        # 转置预测结果，使其形状与target兼容
        # pred形状从(batch_size, 1)变为(1, batch_size)
        pred = pred.t()
        
        # 计算预测是否正确
        # target.reshape(1, -1): 将target从(batch_size,)重塑为(1, batch_size)
        # expand_as(pred): 扩展到与pred相同的形状
        # pred.eq(...): 比较预测结果与真实标签是否相等，返回布尔张量
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        
        # 累积正确预测的数量
        # torch.sum(correct): 统计当前批次中正确预测的样本数
        self.correct += torch.sum(correct)
        
        # 累积总样本数量
        self.total += batch_size

    def compute(self):
        """
        计算最终的准确率
        
        基于累积的统计信息计算全局准确率
        
        @return: torch.Tensor, 准确率值，范围[0, 1]
        """
        # 计算准确率：正确预测数 / 总样本数
        # 转换为float类型确保除法运算的精度
        return self.correct.float() / self.total


if __name__ == "__main__":
    # =============================== 主程序：演示MyAccuracy的使用 ===============================
    
    # 导入工具函数
    from my_utils import setup_seed
    
    # 设置随机种子，确保实验可重复性
    setup_seed(40)
    
    # 重新导入torch（虽然已经在文件开头导入，这里可能是为了演示）
    import torch

    # 创建准确率指标实例
    metric = MyAccuracy()
    
    # 设置批次数
    n_batches = 3
    
    # 模拟多个批次的训练过程
    for i in range(n_batches):
        # 生成随机预测结果
        # torch.randn(10, 5): 生成10个样本，每个样本5个类别的随机值
        # softmax(dim=-1): 在最后一个维度上应用softmax，转换为概率分布
        preds = torch.randn(10, 5).softmax(dim=-1)
        
        # 生成随机真实标签
        # torch.randint(5, (10,)): 生成10个0-4之间的随机整数作为标签
        target = torch.randint(5, (10,))
        
        # 计算当前批次的准确率
        # metric(preds, target) 会调用__call__方法，最终调用.compute()
        # 注意：这里每次调用都会计算累积到当前批次的准确率
        acc = metric(preds, target)
        
        # 打印当前批次的准确率
        print(f"Accuracy on batch {i}: {acc}")

    # 计算所有数据的全局准确率
    acc_avg = metric.compute()
    print(f"Accuracy on all data: {acc_avg}")
    
    # 重置指标状态，准备下一轮计算
    metric.reset()





