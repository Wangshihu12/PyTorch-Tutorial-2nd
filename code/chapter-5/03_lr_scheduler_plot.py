# -*- coding:utf-8 -*-
"""
@file name  : 03_lr_scheduler_plot.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-01
@brief      : scheduler 绘图
@description: 演示PyTorch中各种学习率调度器的使用方法和效果可视化
             包括StepLR、MultiStepLR、ExponentialLR、CosineAnnealingLR、
             ReduceLROnPlateau和LambdaLR等常用调度器
             通过matplotlib绘制学习率变化曲线，直观展示不同调度策略的效果
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可重现
torch.manual_seed(1)

# 全局参数设置
LR = 0.1          # 初始学习率
iteration = 10    # 每个epoch内的迭代次数
max_epoch = 200   # 最大训练轮数

# ------------------------------ 模拟数据和优化器设置 ------------------------------
# 创建模拟的训练数据：一个需要梯度的权重参数和一个目标值

weights = torch.randn((1), requires_grad=True)  # 随机初始化权重，需要计算梯度
target = torch.zeros((1))                       # 目标值设为0

# 创建SGD优化器，设置学习率和动量
optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# ------------------------------ 1. StepLR学习率调度器 ------------------------------
# StepLR：每隔固定步数将学习率乘以gamma
# 适用场景：训练过程中需要定期降低学习率的情况

scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
# step_size=50: 每50个epoch调整一次学习率
# gamma=0.1: 学习率衰减因子，每次调整后学习率变为原来的0.1倍

# 记录学习率变化用于绘图
lr_list, epoch_list = list(), list()

for epoch in range(max_epoch):
    # 获取当前学习率，新版本用get_last_lr()函数，旧版本用get_last_lr()函数，具体看UserWarning
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)

    # 模拟每个epoch内的训练过程
    for i in range(iteration):
        # 计算损失：权重与目标的平方差
        loss = torch.pow((weights - target), 2)
        loss.backward()  # 反向传播计算梯度

        optimizer.step()      # 更新参数
        optimizer.zero_grad() # 清零梯度

    # 每个epoch结束后调整学习率
    scheduler_lr.step()

# 绘制StepLR的学习率变化曲线
plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()

# ------------------------------ 2. MultiStepLR学习率调度器 ------------------------------
# MultiStepLR：在指定的epoch点降低学习率
# 适用场景：知道模型性能瓶颈的具体epoch位置，在这些点精确调整学习率

milestones = [50, 125, 160]  # 指定降低学习率的epoch点
scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
# milestones=[50, 125, 160]: 在第50、125、160个epoch降低学习率
# gamma=0.1: 每次降低学习率变为原来的0.1倍

lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)

    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    scheduler_lr.step()

# 绘制MultiStepLR的学习率变化曲线
plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()

# ------------------------------ 3. ExponentialLR学习率调度器 ------------------------------
# ExponentialLR：每个epoch将学习率乘以gamma
# 适用场景：需要平滑、连续降低学习率的情况

gamma = 0.95  # 学习率衰减因子
scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
# gamma=0.95: 每个epoch学习率变为原来的0.95倍

lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)

    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    scheduler_lr.step()

# 绘制ExponentialLR的学习率变化曲线
plt.plot(epoch_list, lr_list, label="Exponential LR Scheduler\ngamma:{}".format(gamma))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()

# ------------------------------ 4. CosineAnnealingLR学习率调度器 ------------------------------
# CosineAnnealingLR：学习率按余弦函数周期性变化
# 适用场景：需要周期性调整学习率，避免局部最优解

t_max = 50  # 余弦周期长度
scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
# T_max=50: 余弦周期为50个epoch
# eta_min=0: 最小学习率为0

lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)

    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    scheduler_lr.step()

# 绘制CosineAnnealingLR的学习率变化曲线
plt.plot(epoch_list, lr_list, label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()

# ------------------------------ 5. ReduceLROnPlateau学习率调度器 ------------------------------
# ReduceLROnPlateau：当指标停止改善时降低学习率
# 适用场景：根据验证损失或准确率等指标动态调整学习率

# 模拟训练指标
loss_value = 0.5   # 模拟损失值
accuray = 0.9      # 模拟准确率

# 调度器参数设置
factor = 0.1        # 学习率衰减因子
mode = "min"        # 监控模式："min"表示监控损失下降，"max"表示监控准确率上升
patience = 10       # 容忍期：连续10个epoch指标无改善则降低学习率
cooldown = 10       # 冷却期：降低学习率后等待10个epoch再监控
min_lr = 1e-4      # 最小学习率下限
verbose = True      # 是否打印学习率调整信息

# 创建ReduceLROnPlateau调度器
scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                    cooldown=cooldown, min_lr=min_lr, verbose=verbose)

for epoch in range(max_epoch):
    for i in range(iteration):
        # 模拟训练过程
        # train(...)

        optimizer.step()
        optimizer.zero_grad()

    # 模拟第5个epoch时损失值改善
    if epoch == 5:
        loss_value = 0.4

    # 根据损失值调整学习率
    scheduler_lr.step(loss_value)

# ------------------------------ 6. LambdaLR学习率调度器 ------------------------------
# LambdaLR：使用自定义函数动态调整学习率
# 适用场景：需要复杂、灵活的学习率调整策略

lr_init = 0.1  # 初始学习率

# 创建两个不同形状的权重参数
weights_1 = torch.randn((6, 3, 5, 5))  # 卷积层权重
weights_2 = torch.ones((5, 5))          # 全连接层权重

# 创建包含两个参数组的优化器
optimizer = optim.SGD([
    {'params': [weights_1]},  # 第一个参数组：卷积层权重
    {'params': [weights_2]}   # 第二个参数组：全连接层权重
], lr=lr_init)

# 定义两个不同的学习率调整函数
lambda1 = lambda epoch: 0.1 ** (epoch // 20)  # 每20个epoch学习率变为原来的0.1倍
lambda2 = lambda epoch: 0.95 ** epoch         # 每个epoch学习率变为原来的0.95倍

# 创建LambdaLR调度器，为不同参数组应用不同的调整函数
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    for i in range(iteration):
        # 模拟训练过程
        # train(...)

        optimizer.step()
        optimizer.zero_grad()

    # 调整学习率
    scheduler.step()

    # 记录学习率变化
    lr_list.append(scheduler.get_last_lr())
    epoch_list.append(epoch)

    # 打印每个epoch的学习率
    print('epoch:{:5d}, lr:{}'.format(epoch, scheduler.get_last_lr()))

# 绘制LambdaLR的学习率变化曲线：两个参数组分别用不同颜色显示
plt.plot(epoch_list, [i[0] for i in lr_list], label="lambda 1")  # 第一个参数组的学习率
plt.plot(epoch_list, [i[1] for i in lr_list], label="lambda 2")  # 第二个参数组的学习率
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("LambdaLR")
plt.legend()
plt.show()