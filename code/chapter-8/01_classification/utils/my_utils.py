# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import random
import numpy as np
import os
import time

import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from datetime import datetime
import logging
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 原版16
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 原版32
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 原版64
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)

def resnet20():
    """
    https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """
    return ResNet(BasicBlock, [3, 3, 3])

def show_conf_mat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False, save=True):
    """
    混淆矩阵可视化函数：绘制并保存混淆矩阵图片
    
    @param confusion_mat: nd.array, 混淆矩阵数据
    @param classes: list or tuple, 类别名称列表
    @param set_name: str, 数据集名称（train/valid/test）
    @param out_dir: str, 图片保存的文件夹路径
    @param epoch: int, 当前epoch数，用于标题显示
    @param verbose: bool, 是否打印详细的精度信息（召回率和精确率）
    @param perc: bool, 是否使用百分比显示，适用于类别数量很大的情况（如图像分割）
    @param save: bool, 是否保存图片到文件
    @return: matplotlib.figure.Figure, 返回绘制的图形对象
    """
    # 获取类别数量
    cls_num = len(classes)

    # 混淆矩阵归一化处理：将每行除以该行的总和，得到每类的预测概率分布
    confusion_mat_tmp = confusion_mat.copy()  # 创建副本避免修改原始数据
    for i in range(len(classes)):
        # 对每一行进行归一化：预测为该类的样本数 / 该类真实样本总数
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 根据类别数量动态设置图像大小
    if cls_num < 10:
        figsize = 6  # 类别少时使用较小尺寸
    elif cls_num >= 100:
        figsize = 30  # 类别多时使用较大尺寸
    else:
        # 类别数量在10-100之间时，线性插值计算合适的图像大小
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    # 创建matplotlib图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 设置颜色映射：使用灰度色彩映射显示混淆矩阵
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色选项: http://matplotlib.org/examples/color/colormaps_reference.html
    # 绘制混淆矩阵热力图
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    # 添加颜色条，显示数值与颜色的对应关系
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')  # 设置颜色条刻度字体大小

    # 设置坐标轴标签和刻度
    xlocations = np.array(range(len(classes)))  # 创建类别索引数组
    # 设置x轴（预测标签）
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # 旋转标签避免重叠
    # 设置y轴（真实标签）
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    # 设置坐标轴标题
    ax.set_xlabel('预测标签')  # 预测标签
    ax.set_ylabel('真实标签')     # 真实标签
    # 设置图形标题
    ax.set_title("混淆矩阵_{}_{}".format(set_name, epoch))

    # 在混淆矩阵中显示数值
    if perc:
        # 百分比模式：计算每个预测类别的百分比
        cls_per_nums = confusion_mat.sum(axis=0)  # 每个预测类别的总样本数
        conf_mat_per = confusion_mat / cls_per_nums  # 计算百分比
        # 在每个格子中显示百分比
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), 
                       va='center', ha='center', color='red', fontsize=10)
    else:
        # 绝对数值模式：显示原始的样本数量
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), 
                       va='center', ha='center', color='red', fontsize=10)
    
    # 保存图片到指定目录
    if save:
        fig.savefig(os.path.join(out_dir, "混淆矩阵_{}.png".format(set_name)))
    plt.close()  # 关闭图形释放内存

    # 详细精度信息输出：计算并显示每个类别的召回率和精确率
    if verbose:
        for i in range(cls_num):
            # 计算召回率：正确预测数 / 真实样本总数
            recall = confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :]))
            # 计算精确率：正确预测数 / 预测为该类的总数
            precision = confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))
            # 打印每个类别的详细信息
            print('类别:{:<10}, 总样本数:{:<6}, 正确数:{:<5}  召回率: {:.2%} 精确率: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i], recall, precision))

    return fig  # 返回图形对象，可用于进一步处理或显示


class ModelTrainer(object):
    """
    模型训练器类：提供模型训练和评估的静态方法
    包含单epoch训练和模型评估两个核心功能
    """

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger, classes):
        """
        训练一个完整的epoch
        
        @param data_loader: 数据加载器，提供训练数据批次
        @param model: 要训练的深度学习模型
        @param loss_f: 损失函数
        @param optimizer: 优化器，用于更新模型参数
        @param scheduler: 学习率调度器
        @param epoch_idx: 当前epoch索引
        @param device: 计算设备（GPU或CPU）
        @param args: 训练参数配置
        @param logger: 日志记录器
        @param classes: 类别标签列表
        @return: 返回训练损失、Top1准确率和混淆矩阵
        """
        # 将模型设置为训练模式，启用dropout、batch normalization等训练相关层
        model.train()
        # 记录训练开始时间
        end = time.time()

        # 获取类别数量，用于初始化混淆矩阵
        class_num = len(classes)
        # 初始化混淆矩阵：行表示真实标签，列表示预测标签
        conf_mat = np.zeros((class_num, class_num))

        # 创建各种指标的平均值计算器
        loss_m = AverageMeter()      # 损失值统计器
        top1_m = AverageMeter()      # Top1准确率统计器
        top5_m = AverageMeter()      # Top5准确率统计器
        batch_time_m = AverageMeter() # 批次处理时间统计器

        # 计算最后一个批次的索引，用于日志显示
        last_idx = len(data_loader) - 1
        
        # 遍历数据加载器中的每个批次
        for batch_idx, data in enumerate(data_loader):

            # 解包数据：inputs为输入图像，labels为真实标签
            inputs, labels = data
            # 将数据移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播：模型推理，计算输出
            outputs = model(inputs)
            # 清空梯度：防止梯度累积
            optimizer.zero_grad()

            # 计算损失：将输出和标签移到CPU上计算损失（某些损失函数可能不支持GPU）
            loss = loss_f(outputs.cpu(), labels.cpu())
            # 反向传播：计算梯度
            loss.backward()
            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 计算准确率：Top1和Top5准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # 获取预测结果：选择概率最高的类别作为预测标签
            _, predicted = torch.max(outputs.data, 1)
            
            # 更新混淆矩阵：统计每个类别的预测情况
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实类别索引
                pre_i = predicted[j].cpu().numpy()  # 预测类别索引
                conf_mat[cate_i, pre_i] += 1.       # 在对应位置累加计数

            # 更新统计指标：记录当前批次的损失、准确率等指标
            # 注意：update方法内部使用 self.sum += val * n，因此需要传入batch数量
            loss_m.update(loss.item(), inputs.size(0))      # 更新损失统计
            top1_m.update(acc1.item(), outputs.size(0))    # 更新Top1准确率统计
            top5_m.update(acc5.item(), outputs.size(0))    # 更新Top5准确率统计

            # 更新批次处理时间统计
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # 按指定频率打印训练信息：每隔print_freq个批次打印一次
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, top1=top1_m, top5=top5_m))  # val是当次传进去的值，avg是整体平均值。
        
        # 返回训练结果：平均损失、Top1准确率和混淆矩阵
        return loss_m, top1_m, conf_mat

    @staticmethod
    def evaluate(data_loader, model, loss_f, device, classes):
        """
        评估模型性能：在验证集或测试集上评估模型
        
        @param data_loader: 数据加载器，提供评估数据批次
        @param model: 要评估的深度学习模型
        @param loss_f: 损失函数
        @param device: 计算设备（GPU或CPU）
        @param classes: 类别标签列表
        @return: 返回评估损失、Top1准确率和混淆矩阵
        """
        # 将模型设置为评估模式，禁用dropout、batch normalization等训练相关层
        model.eval()

        # 获取类别数量，用于初始化混淆矩阵
        class_num = len(classes)
        # 初始化混淆矩阵：行表示真实标签，列表示预测标签
        conf_mat = np.zeros((class_num, class_num))

        # 创建各种指标的平均值计算器
        loss_m = AverageMeter()      # 损失值统计器
        top1_m = AverageMeter()      # Top1准确率统计器
        top5_m = AverageMeter()      # Top5准确率统计器

        # 遍历数据加载器中的每个批次
        for i, data in enumerate(data_loader):
            # 解包数据：inputs为输入图像，labels为真实标签
            inputs, labels = data
            # 将数据移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播：模型推理，计算输出（不计算梯度）
            outputs = model(inputs)
            # 计算损失：将输出和标签移到CPU上计算损失
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 计算准确率：Top1和Top5准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # 获取预测结果：选择概率最高的类别作为预测标签
            _, predicted = torch.max(outputs.data, 1)
            
            # 更新混淆矩阵：统计每个类别的预测情况
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实类别索引
                pre_i = predicted[j].cpu().numpy()  # 预测类别索引
                conf_mat[cate_i, pre_i] += 1.       # 在对应位置累加计数

            # 更新统计指标：记录当前批次的损失、准确率等指标
            # 注意：update方法内部使用 self.sum += val * n，因此需要传入batch数量
            loss_m.update(loss.item(), inputs.size(0))      # 更新损失统计
            top1_m.update(acc1.item(), outputs.size(0))    # 更新Top1准确率统计
            top5_m.update(acc5.item(), outputs.size(0))    # 更新Top5准确率统计

        # 返回评估结果：平均损失、Top1准确率和混淆矩阵
        return loss_m, top1_m, conf_mat


class ModelTrainerEnsemble(ModelTrainer):
    @staticmethod
    def average(outputs):
        """Compute the average over a list of tensors with the same size."""
        return sum(outputs) / len(outputs)

    @staticmethod
    def evaluate(data_loader, models, loss_f, device, classes):

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        top1_m = torchmetrics.Accuracy().to(device)

        # top1 acc group
        top1_group = []
        for model_idx in range(len(models)):
            top1_group.append(torchmetrics.Accuracy().to(device))

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = []
            for model_idx, model in enumerate(models):
                output_single = F.softmax(model(inputs), dim=1)
                outputs.append(output_single)
                # 计算单个模型acc
                top1_group[model_idx](output_single, labels)
                # 计算单个模型loss

            # 计算acc 组
            output_avg = ModelTrainerEnsemble.average(outputs)
            top1_m(output_avg, labels)

            # loss 组
            loss = loss_f(output_avg.cpu(), labels.cpu())
            loss_m.update(loss.item(), inputs.size(0))

        return loss_m, top1_m.compute(), top1_group, conf_mat


class Logger(object):
    """
    自定义日志记录器类：提供文件和控制台双重输出功能
    
    功能特点：
    1. 同时输出到文件和控制台
    2. 自动创建日志目录
    3. 可配置的日志格式和级别
    4. 支持多个Handler
    """
    
    def __init__(self, path_log):
        """
        初始化Logger实例
        
        @param path_log: 日志文件的完整路径
        """
        # 从完整路径中提取日志文件名（不包含路径）
        log_name = os.path.basename(path_log)
        # 设置日志器名称：如果有文件名则使用，否则使用"root"
        self.log_name = log_name if log_name else "root"
        # 保存日志文件的完整输出路径
        self.out_path = path_log

        # 获取日志文件所在的目录路径
        log_dir = os.path.dirname(self.out_path)
        # 检查日志目录是否存在，如果不存在则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        """
        初始化并配置日志记录器
        
        配置内容：
        1. 设置日志级别为INFO
        2. 创建文件输出Handler
        3. 创建控制台输出Handler
        4. 配置日志格式
        5. 将Handler添加到logger
        
        @return: 配置好的logging.Logger实例
        """
        # 获取或创建指定名称的logger实例
        logger = logging.getLogger(self.log_name)
        # 设置logger的日志级别为INFO（记录INFO及以上级别的日志）
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler：将日志输出到文件
        # 创建FileHandler，'w'模式表示覆盖写入（每次运行清空之前的日志）
        file_handler = logging.FileHandler(self.out_path, 'w')
        # 设置文件Handler的日志级别为INFO
        file_handler.setLevel(logging.INFO)
        # 创建日志格式器：时间 - 日志器名称 - 日志级别 - 日志消息
        # 示例输出：2023-09-25 14:30:45,123 - root - INFO - 开始训练
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 将格式器应用到文件Handler
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler：将日志输出到控制台
        # 创建StreamHandler，默认输出到sys.stdout（控制台）
        console_handler = logging.StreamHandler()
        # 设置控制台Handler的日志级别为INFO
        console_handler.setLevel(logging.INFO)
        # 注释掉的代码：为控制台Handler设置格式器
        # 如果不设置，控制台输出将使用默认格式
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 将配置好的Handler添加到logger
        # 这样logger就可以同时输出到文件和控制台
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 返回配置好的logger实例
        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    
    @param out_dir: 输出目录路径，日志文件夹将在此目录下创建
    @return: 返回两个值：(logger, log_dir)
             - logger: 配置好的日志记录器对象
             - log_dir: 新创建的日志文件夹的完整路径
    """
    # 获取当前时间
    now_time = datetime.now()
    
    # 将当前时间格式化为字符串：年-月-日_时-分-秒
    # 例如：2023-09-25_14-30-45
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    
    # 构建日志文件夹的完整路径：输出目录 + 时间字符串
    # 根据config中的创建时间作为文件夹名，确保每次运行都有唯一的日志目录
    log_dir = os.path.join(out_dir, time_str)
    
    # 检查日志文件夹是否存在，如果不存在则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 递归创建目录，包括父目录
    
    # 创建logger实例
    # 构建日志文件的完整路径：日志目录 + 日志文件名
    path_log = os.path.join(log_dir, "log.log")
    
    # 创建Logger类的实例
    logger = Logger(path_log)
    
    # 初始化日志记录器，配置日志格式、级别等参数
    logger = logger.init_logger()
    
    # 返回配置好的日志记录器和日志目录路径
    return logger, log_dir


def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


class AverageMeter:
    """Computes and stores the average and current value
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]