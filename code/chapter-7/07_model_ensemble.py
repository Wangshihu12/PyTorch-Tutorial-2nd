# -*- coding:utf-8 -*-
"""
@file name  : 07_model_ensemble.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-30
@brief      : 模型集成 - 演示多种集成学习方法的性能对比
             
             本代码实现了6种不同的集成学习方法，使用相同的LeNet5基础模型和CIFAR-10数据集
             对比不同集成策略在准确率、训练时间和评估时间上的表现
             为选择合适的集成方法提供实验依据
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入torchensemble框架提供的各种集成分类器
from torchensemble.fusion import FusionClassifier              # 融合分类器
from torchensemble.voting import VotingClassifier             # 投票分类器
from torchensemble.bagging import BaggingClassifier           # Bagging分类器
from torchensemble.gradient_boosting import GradientBoostingClassifier  # 梯度提升分类器
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier  # 快照集成分类器
from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier  # 软梯度提升分类器

# 导入日志设置工具
from torchensemble.utils.logging import set_logger


def display_records(records, logger):
    """
    显示不同集成方法的性能记录
    
    以表格形式展示各种集成方法的测试准确率、训练时间和评估时间
    
    @param records: list, 包含(方法名, 训练时间, 评估时间, 准确率)的元组列表
    @param logger: Logger, 日志记录器
    """
    # 定义输出格式：方法名 | 测试准确率 | 训练时间 | 评估时间
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")  # 打印空行，分隔输出
    
    # 遍历所有记录，格式化输出
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))


class LeNet5(nn.Module):
    """
    LeNet5卷积神经网络
    
    经典的LeNet5架构，适用于图像分类任务
    网络结构：卷积层 -> 池化层 -> 卷积层 -> 池化层 -> 全连接层 -> 全连接层 -> 输出层
    
    特点：
    1. 轻量级网络，训练速度快
    2. 适合小尺寸图像（如32x32）
    3. 经典的CNN架构设计
    """
    
    def __init__(self):
        """
        初始化LeNet5网络结构
        
        定义各层的参数和连接关系
        """
        super(LeNet5, self).__init__()
        
        # 第一个卷积层：输入3通道，输出6通道，卷积核5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # 最大池化层：2x2池化窗口，步长2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 第二个卷积层：输入6通道，输出16通道，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # 第一个全连接层：输入400维，输出120维
        # 400 = 16 * 5 * 5（第二个卷积层的输出特征图尺寸）
        self.fc1 = nn.Linear(400, 120)
        
        # 第二个全连接层：输入120维，输出84维
        self.fc2 = nn.Linear(120, 84)
        
        # 输出层：输入84维，输出10维（CIFAR-10的类别数）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播函数
        
        @param x: torch.Tensor, 输入图像张量，形状为(batch_size, 3, 32, 32)
        @return: torch.Tensor, 输出预测结果，形状为(batch_size, 10)
        """
        # 第一个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))  # 输出形状：(batch_size, 6, 14, 14)
        
        # 第二个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出形状：(batch_size, 16, 5, 5)
        
        # 展平特征图，准备输入全连接层
        x = x.view(-1, 400)  # 形状变为：(batch_size, 400)
        
        # 第一个全连接层 + ReLU激活
        x = F.relu(self.fc1(x))  # 输出形状：(batch_size, 120)
        
        # 第二个全连接层 + ReLU激活
        x = F.relu(self.fc2(x))  # 输出形状：(batch_size, 84)
        
        # 输出层（无激活函数，用于计算损失）
        x = self.fc3(x)  # 输出形状：(batch_size, 10)
        
        return x


if __name__ == "__main__":
    # =============================== 主程序：集成方法性能对比 ===============================
    
    # =============================== 超参数设置 ===============================
    n_estimators = 5      # 集成中模型的数量
    lr = 1e-3             # 学习率
    weight_decay = 5e-4   # 权重衰减（L2正则化）
    epochs = 100          # 训练轮数

    # =============================== 工具配置 ===============================
    data_dir = r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office"  # 数据目录
    batch_size = 128      # 批次大小
    records = []          # 记录各种方法的性能指标
    torch.manual_seed(0)  # 设置随机种子，确保实验可重复性

    # =============================== 数据加载和预处理 ===============================
    
    # 训练集的数据增强变换
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            transforms.RandomCrop(32, 4),        # 随机裁剪，padding=4
            transforms.ToTensor(),                # 转换为PyTorch张量
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),       # RGB三通道均值
                (0.2023, 0.1994, 0.2010)        # RGB三通道标准差
            ),
        ]
    )

    # 验证集的变换（不需要数据增强）
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),                # 转换为PyTorch张量
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),       # RGB三通道均值
                (0.2023, 0.1994, 0.2010)        # RGB三通道标准差
            ),
        ]
    )

    # 加载CIFAR-10数据集
    # 注意：root变量下需要存放cifar-10-python.tar.gz文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4)

    # 设置日志记录器，启用TensorBoard日志
    logger = set_logger("classification_cifar10_cnn", use_tb_logger=True)
    
    # =============================== 方法1：FusionClassifier（融合分类器） ===============================
    # 融合分类器：将多个模型的输出进行加权融合
    model = FusionClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    tic = time.time()  # 记录训练开始时间
    model.fit(train_loader, epochs=epochs)
    toc = time.time()  # 记录训练结束时间
    training_time = toc - tic  # 计算训练时间

    # 评估阶段
    tic = time.time()  # 记录评估开始时间
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()  # 记录评估结束时间
    evaluating_time = toc - tic  # 计算评估时间

    # 记录性能指标
    records.append(
        ("FusionClassifier", training_time, evaluating_time, testing_acc)
    )

    # =============================== 方法2：VotingClassifier（投票分类器） ===============================
    # 投票分类器：通过多数投票决定最终预测结果
    model = VotingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # 评估阶段
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    # 记录性能指标
    records.append(
        ("VotingClassifier", training_time, evaluating_time, testing_acc)
    )

    # =============================== 方法3：BaggingClassifier（Bagging分类器） ===============================
    # Bagging分类器：通过Bootstrap采样训练多个模型
    model = BaggingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # 评估阶段
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    # 记录性能指标
    records.append(
        ("BaggingClassifier", training_time, evaluating_time, testing_acc)
    )

    # =============================== 方法4：GradientBoostingClassifier（梯度提升分类器） ===============================
    # 梯度提升分类器：通过梯度提升算法训练多个模型
    model = GradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    # 注意：梯度提升通常需要更多时间，这里只训练1个epoch作为演示
    tic = time.time()
    # model.fit(train_loader, epochs=epochs)  # 完整训练
    model.fit(train_loader, epochs=1)         # 快速演示
    toc = time.time()
    training_time = toc - tic

    # 评估阶段
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    # 记录性能指标
    records.append(
        (
            "GradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # =============================== 方法5：SnapshotEnsembleClassifier（快照集成分类器） ===============================
    # 快照集成分类器：在训练过程中保存多个快照模型
    model = SnapshotEnsembleClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # 评估阶段
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    # 记录性能指标
    records.append(
        (
            "SnapshotEnsembleClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # =============================== 方法6：SoftGradientBoostingClassifier（软梯度提升分类器） ===============================
    # 软梯度提升分类器：梯度提升的软版本，使用软标签
    model = SoftGradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # 训练阶段
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # 评估阶段
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    # 记录性能指标
    records.append(
        (
            "SoftGradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # =============================== 结果展示 ===============================
    # 打印不同集成方法的性能对比结果
    display_records(records, logger)


