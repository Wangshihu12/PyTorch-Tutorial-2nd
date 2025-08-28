# -*- coding:utf-8 -*-
"""
@file name  : 03_confusion_matrix.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 混淆矩阵绘制及训练曲线记录
"""
import torch
import numpy as np
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from my_utils import *

# =============================== 设备配置 ===============================
# 自动检测并使用可用的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================== 超参数设置 ===============================
train_bs = 128          # 训练批次大小
valid_bs = 128          # 验证批次大小
lr_init = 0.01          # 初始学习率
max_epoch = 200         # 最大训练轮数
print_interval = 100    # 打印训练信息的间隔（每多少个iteration打印一次）
# CIFAR-10数据集的10个类别名称
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# =============================== 日志和结果保存配置 ===============================
# 设置结果保存目录
result_dir = os.path.join("Result")

# 获取当前时间，用于创建带时间戳的日志目录
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

# 创建带时间戳的日志目录
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建TensorBoard写入器，用于记录训练过程中的各种指标
writer = SummaryWriter(log_dir=log_dir)

# =============================== 步骤1/5：数据加载和预处理 ===============================
# 定义图像标准化参数（CIFAR-10数据集的统计参数）
normMean = [0.4948052, 0.48568845, 0.44682974]    # RGB三个通道的均值
normStd = [0.24580306, 0.24236229, 0.2603115]     # RGB三个通道的标准差

# 创建标准化变换
normTransform = transforms.Normalize(normMean, normStd)

# 训练集的数据增强变换
train_transform = transforms.Compose([
    transforms.Resize(32),                    # 调整图像尺寸为32x32
    transforms.RandomCrop(32, padding=4),    # 随机裁剪，增加数据多样性
    transforms.ToTensor(),                    # 转换为PyTorch张量
    normTransform                             # 应用标准化
])

# 验证集的变换（不需要数据增强）
valid_transform = transforms.Compose([
    transforms.ToTensor(),                    # 转换为PyTorch张量
    normTransform                             # 应用标准化
])

# 构建CIFAR-10数据集实例
# 注意：root变量下需要存放cifar-10-python.tar.gz文件
# cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
data_dir = r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office"
train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

# 构建数据加载器
train_loader = DataLoader(dataset=train_set, batch_size=train_bs, shuffle=True)  # 训练集打乱顺序
valid_loader = DataLoader(dataset=test_set, batch_size=valid_bs)                 # 验证集保持顺序

# =============================== 步骤2/5：定义网络模型 ===============================
# 创建ResNet8模型实例
model = resnet8()
# 将模型移动到指定设备（GPU或CPU）
model.to(device)

# =============================== 步骤3/5：定义损失函数和优化器 ===============================
criterion = nn.CrossEntropyLoss()  # 选择交叉熵损失函数，适用于多分类问题
# 选择SGD优化器，设置动量、阻尼等参数
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
# 设置学习率调度器：每80个epoch将学习率乘以0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

# =============================== 步骤4/5：模型训练 ===============================
# 开始训练循环
for epoch in range(max_epoch):
    # 注意：学习率更新移到了每个epoch结束后
    
    # =============================== 训练阶段 ===============================
    # 初始化训练相关的变量
    class_num = len(classes_name)                    # 类别数量
    conf_mat = np.zeros((class_num, class_num))      # 初始化混淆矩阵
    loss_sigma = []                                  # 存储所有iteration的损失值
    loss_avg = 0                                     # 平均损失
    acc_avg = 0                                      # 平均准确率
    path_error = []                                  # 错误预测的路径（当前未使用）
    label_list = []                                  # 标签列表（当前未使用）
    
    # 设置模型为训练模式
    model.train()
    
    # 遍历训练数据加载器
    for i, data in enumerate(train_loader):
        # 调试用：限制训练iteration数量
        # if i == 30 : break
        
        # 获取输入图像和标签
        inputs, labels = data
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播、反向传播、权重更新
        optimizer.zero_grad()        # 清空梯度
        outputs = model(inputs)      # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()              # 反向传播
        optimizer.step()             # 更新权重

        # 统计预测信息
        loss_sigma.append(loss.item())           # 记录当前iteration的损失
        loss_avg = np.mean(loss_sigma)          # 计算平均损失

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        
        # 更新混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()    # 真实标签
            pre_i = predicted[j].cpu().numpy()  # 预测标签
            conf_mat[cate_i, pre_i] += 1.       # 在混淆矩阵对应位置加1
        
        # 计算当前准确率（对角线元素之和除以总元素之和）
        acc_avg = conf_mat.trace() / conf_mat.sum()

        # 每print_interval个iteration打印一次训练信息
        # loss为当前所有iteration的平均值
        if i % print_interval == print_interval - 1:
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                  format(epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))
    
    # 每个epoch结束后更新学习率
    scheduler.step()

    # =============================== 记录训练指标到TensorBoard ===============================
    # 记录训练损失
    writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
    # 记录当前学习率
    writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
    # 记录训练准确率
    writer.add_scalars('Accuracy_group', {'train_acc': acc_avg}, epoch)

    # 生成并保存训练集混淆矩阵图
    conf_mat_figure = show_conf_mat(conf_mat, classes_name, "train", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    # 将混淆矩阵图添加到TensorBoard中
    writer.add_figure('confusion_matrix_train', conf_mat_figure, global_step=epoch)

    # =============================== 验证阶段 ===============================
    # 观察模型在验证集上的表现
    # 重新初始化验证相关的变量
    class_num = len(classes_name)                    # 类别数量
    conf_mat = np.zeros((class_num, class_num))      # 初始化混淆矩阵
    loss_sigma = []                                  # 存储所有iteration的损失值
    loss_avg = 0                                     # 平均损失
    acc_avg = 0                                      # 平均准确率
    path_error = []                                  # 错误预测的路径（当前未使用）
    label_list = []                                  # 标签列表（当前未使用）

    # 设置模型为评估模式（不计算梯度）
    model.eval()
    
    # 遍历验证数据加载器
    for i, data in enumerate(valid_loader):
        # 获取输入图像和标签
        inputs, labels = data
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播（不计算梯度）
        outputs = model(inputs)
        outputs.detach_()  # 分离梯度，节省内存
        
        # 计算损失
        loss = criterion(outputs, labels)

        # 统计预测信息
        loss_sigma.append(loss.item())           # 记录当前iteration的损失
        loss_avg = np.mean(loss_sigma)          # 计算平均损失

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        
        # 更新混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()    # 真实标签
            pre_i = predicted[j].cpu().numpy()  # 预测标签
            conf_mat[cate_i, pre_i] += 1.       # 在混淆矩阵对应位置加1
        
        # 计算当前准确率
        acc_avg = conf_mat.trace() / conf_mat.sum()

    # 打印验证集准确率
    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
    
    # =============================== 记录验证指标到TensorBoard ===============================
    # 记录验证损失和准确率
    writer.add_scalars('Loss_group', {'valid_loss': loss_avg}, epoch)
    writer.add_scalars('Accuracy_group', {'valid_acc': acc_avg}, epoch)
    
    # 生成并保存验证集混淆矩阵图
    conf_mat_figure = show_conf_mat(conf_mat, classes_name, "valid", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    # 将混淆矩阵图添加到TensorBoard中
    writer.add_figure('confusion_matrix_valid', conf_mat_figure, global_step=epoch)

# =============================== 训练完成 ===============================
print('Finished Training')


