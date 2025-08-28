# -*- coding: utf-8 -*-
"""
# @file name  : 02_fintune-freeze.py
# @author     :  TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-24
# @brief      : finetune方法之冻结特征提取层
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from my_utils import *

# =============================== 基础配置 ===============================
# 获取当前文件所在目录的绝对路径
BASEDIR = os.path.dirname(os.path.abspath(__file__))

# 自动检测并使用可用的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

# 定义类别标签映射：蚂蚁为0，蜜蜂为1
label_name = {"ants": 0, "bees": 1}


class AntsDataset(Dataset):
    """
    蚂蚁蜜蜂数据集类，继承自PyTorch的Dataset类
    
    用于加载和处理蚂蚁蜜蜂图像数据，支持从文件夹结构自动推断标签
    """
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        
        @param data_dir: str, 数据集的根目录路径，子文件夹名称表示类别
        @param transform: 图像预处理变换，默认为None
        """
        self.label_name = {"ants": 0, "bees": 1}  # 类别名称到索引的映射
        self.data_info = self.get_img_info(data_dir)  # 获取所有图像的信息
        self.transform = transform  # 图像预处理变换

    def __getitem__(self, index):
        """
        根据索引获取单个数据样本
        
        @param index: int, 数据样本的索引
        @return: tuple, 返回(图像张量, 标签)的元组
        """
        # 根据索引获取图像路径和标签
        path_img, label = self.data_info[index]
        # 打开图像并转换为RGB格式
        img = Image.open(path_img).convert('RGB')

        # 如果定义了预处理变换，则应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        获取数据集的总样本数量
        
        @return: int, 数据集中样本的总数
        """
        return len(self.data_info)

    def get_img_info(self, data_dir):
        """
        从文件夹结构中读取图像路径和标签信息
        
        @param data_dir: str, 数据目录路径
        @return: list, 包含(图像路径, 标签)元组的列表
        """
        data_info = list()
        
        # 遍历根目录下的所有子目录和文件
        for root, dirs, _ in os.walk(data_dir):
            # 遍历每个类别子目录
            for sub_dir in dirs:
                # 获取该类别下的所有图像文件名
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 过滤出JPG格式的图像文件
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历该类别下的所有图像
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 构建完整的图像文件路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 根据子目录名称获取标签
                    label = self.label_name[sub_dir]
                    # 将图像路径和标签添加到列表中
                    data_info.append((path_img, int(label)))

        # 检查数据集是否为空
        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

# =============================== 超参数设置 ===============================
max_epoch = 25          # 最大训练轮数
BATCH_SIZE = 16         # 批次大小
LR = 0.001              # 初始学习率
log_interval = 10       # 日志记录间隔
val_interval = 1        # 验证间隔
classes = 2             # 分类类别数（蚂蚁和蜜蜂）
start_epoch = -1        # 起始训练轮数（-1表示从0开始）
lr_decay_step = 7       # 学习率衰减步长
print_interval = 2      # 打印训练信息的间隔

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

# =============================== 步骤1/5：数据准备 ===============================
# 数据集下载链接：https://download.pytorch.org/tutorial/hymenoptera_data.zip
data_dir = r"G:\deep_learning_data\hymenoptera_data"

# 设置训练集和验证集目录
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

# 定义图像标准化参数（ImageNet数据集的统计参数）
norm_mean = [0.485, 0.456, 0.406]    # RGB三个通道的均值
norm_std = [0.229, 0.224, 0.225]     # RGB三个通道的标准差

# 训练集的数据增强变换
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),     # 随机裁剪并调整到224x224
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),                 # 转换为PyTorch张量
    transforms.Normalize(norm_mean, norm_std),  # 应用标准化
])

# 验证集的变换（不需要数据增强）
valid_transform = transforms.Compose([
    transforms.Resize(256),                # 调整图像尺寸为256x256
    transforms.CenterCrop(224),           # 中心裁剪到224x224
    transforms.ToTensor(),                 # 转换为PyTorch张量
    transforms.Normalize(norm_mean, norm_std),  # 应用标准化
])

# 构建数据集实例
train_data = AntsDataset(data_dir=train_dir, transform=train_transform)
valid_data = AntsDataset(data_dir=valid_dir, transform=valid_transform)

# 构建数据加载器
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # 训练集打乱顺序
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)                 # 验证集保持顺序

# =============================== 步骤2/5：模型构建和微调 ===============================

# 1/3 构建ResNet18模型
resnet18_ft = models.resnet18()

# 2/3 加载预训练参数
# 下载预训练模型：https://download.pytorch.org/models/resnet18-f37072fd.pth
path_pretrained_model = r"F:\pytorch-tutorial-2nd\data\model_zoo\resnet18-f37072fd.pth"
state_dict_load = torch.load(path_pretrained_model)
resnet18_ft.load_state_dict(state_dict_load)

# 3/3 冻结卷积层参数（微调的核心策略）
# 方法1：冻结所有卷积层参数，只训练全连接层
for param in resnet18_ft.parameters():
    param.requires_grad = False  # 设置参数不需要梯度，即不参与训练

# 打印第一个卷积层的权重，验证冻结是否成功
print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))

# 替换全连接层，适配新的分类任务
num_ftrs = resnet18_ft.fc.in_features  # 获取原全连接层的输入特征数
resnet18_ft.fc = nn.Linear(num_ftrs, classes)  # 创建新的全连接层，输出类别数为2

# 将模型移动到指定设备
resnet18_ft.to(device)

# =============================== 步骤3/5：损失函数定义 ===============================
# 选择交叉熵损失函数，适用于多分类问题
criterion = nn.CrossEntropyLoss()

# =============================== 步骤4/5：优化器和学习率调度器 ===============================
# 选择SGD优化器，设置学习率和动量
optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)

# 设置学习率调度器：每7个epoch将学习率乘以0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

# =============================== 步骤5/5：模型训练 ===============================
# 创建列表用于存储训练和验证曲线
train_curve = list()
valid_curve = list()

# 开始训练循环
for epoch in range(start_epoch + 1, max_epoch):
    # =============================== 训练阶段 ===============================
    # 初始化训练相关的变量
    class_num = classes                    # 类别数量
    conf_mat = np.zeros((class_num, class_num))      # 初始化混淆矩阵
    loss_sigma = []                                  # 存储所有iteration的损失值
    loss_avg = 0                                     # 平均损失
    acc_avg = 0                                      # 平均准确率
    path_error = []                                  # 错误预测的路径（当前未使用）
    label_list = []                                  # 标签列表（当前未使用）

    # 设置模型为训练模式
    resnet18_ft.train()
    
    # 遍历训练数据加载器
    for i, data in enumerate(train_loader):
        # 获取输入图像和标签
        inputs, labels = data
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播、反向传播、权重更新
        optimizer.zero_grad()        # 清空梯度
        outputs = resnet18_ft(inputs)      # 前向传播
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
        
        # 计算当前准确率
        acc_avg = conf_mat.trace() / conf_mat.sum()

        # 每print_interval个iteration打印一次训练信息
        if i % print_interval == print_interval - 1:
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                  format(epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))

            # 打印第一个卷积层的权重，验证冻结是否成功
            print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

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
    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "train", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    # 将混淆矩阵图添加到TensorBoard中
    writer.add_figure('confusion_matrix_train', conf_mat_figure, global_step=epoch)

    # =============================== 验证阶段 ===============================
    # 重新初始化验证相关的变量
    class_num = classes                    # 类别数量
    conf_mat = np.zeros((class_num, class_num))      # 初始化混淆矩阵
    loss_sigma = []                                  # 存储所有iteration的损失值
    loss_avg = 0                                     # 平均损失
    acc_avg = 0                                      # 平均准确率
    path_error = []                                  # 错误预测的路径（当前未使用）
    label_list = []                                  # 标签列表（当前未使用）

    # 设置模型为评估模式
    resnet18_ft.eval()
    
    # 在验证阶段不计算梯度，节省内存
    with torch.no_grad():
        # 遍历验证数据加载器
        for j, data in enumerate(valid_loader):
            # 获取输入图像和标签
            inputs, labels = data
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播（不计算梯度）
            outputs = resnet18_ft(inputs)
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
    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "valid", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    # 将混淆矩阵图添加到TensorBoard中
    writer.add_figure('confusion_matrix_valid', conf_mat_figure, global_step=epoch)

# =============================== 训练完成 ===============================
print('Finished Training')


