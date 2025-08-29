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
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')

    # 设置文字
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # , fontsize='small'
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if save:
        fig.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))

    return fig


class ModelTrainer(object):
    """
    模型训练器类
    
    提供模型训练和评估的静态方法，包含完整的训练循环和性能评估逻辑
    使用静态方法设计，便于调用而不需要实例化
    """

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger, classes):
        """
        训练一个完整的epoch
        
        执行完整的训练循环，包括前向传播、反向传播、参数更新和性能监控
        
        @param data_loader: DataLoader, 训练数据加载器
        @param model: nn.Module, 要训练的神经网络模型
        @param loss_f: nn.Module, 损失函数
        @param optimizer: torch.optim.Optimizer, 优化器
        @param scheduler: torch.optim.lr_scheduler._LRScheduler, 学习率调度器
        @param epoch_idx: int, 当前epoch的索引
        @param device: torch.device, 计算设备（GPU或CPU）
        @param args: argparse.Namespace, 训练参数配置
        @param logger: Logger, 日志记录器
        @param classes: list, 类别名称列表
        
        @return: tuple, 返回(损失统计器, Top1准确率统计器, 混淆矩阵)
        """
        # 设置模型为训练模式（启用dropout、batch normalization等）
        model.train()
        
        # 记录epoch开始时间
        end = time.time()

        # 获取类别数量
        class_num = len(classes)
        
        # 初始化混淆矩阵，用于统计分类结果
        # conf_mat[i][j] 表示真实类别为i，预测类别为j的样本数量
        conf_mat = np.zeros((class_num, class_num))

        # 创建各种指标的统计器
        loss_m = AverageMeter()        # 损失值统计器
        top1_m = AverageMeter()        # Top1准确率统计器
        top5_m = AverageMeter()        # Top5准确率统计器
        batch_time_m = AverageMeter()  # 批次处理时间统计器

        # 获取最后一个批次的索引，用于进度显示
        last_idx = len(data_loader) - 1
        
        # 遍历训练数据的所有批次
        for batch_idx, data in enumerate(data_loader):
            # 解包数据，获取输入和标签
            inputs, labels = data
            
            # 将数据移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            
            # =============================== 前向传播和反向传播 ===============================
            # 前向传播：计算模型输出
            outputs = model(inputs)
            
            # 清空梯度缓存，避免梯度累积
            optimizer.zero_grad()

            # 计算损失（注意：将输出和标签移到CPU上计算损失）
            # 这是因为某些损失函数在GPU上可能不稳定
            loss = loss_f(outputs.cpu(), labels.cpu())
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # =============================== 性能计算 ===============================
            # 计算Top1和Top5准确率
            # acc1: Top1准确率，即预测概率最高的类别是否正确
            # acc5: Top5准确率，即真实标签是否在预测概率前5的类别中
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # 获取预测结果（Top1预测）
            # torch.max返回最大值和对应的索引，这里我们只需要索引
            _, predicted = torch.max(outputs.data, 1)
            
            # 更新混淆矩阵
            # 统计每个类别的预测情况
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实类别索引
                pre_i = predicted[j].cpu().numpy()  # 预测类别索引
                conf_mat[cate_i, pre_i] += 1.       # 在对应位置加1

            # =============================== 指标记录 ===============================
            # 更新各种统计指标
            # 注意：update方法的第二个参数是样本数量，用于计算加权平均
            # 因为update里：self.sum += val * n，所以需要传入batch数量
            loss_m.update(loss.item(), inputs.size(0))      # 更新损失统计
            top1_m.update(acc1.item(), outputs.size(0))    # 更新Top1准确率统计
            top5_m.update(acc5.item(), outputs.size(0))    # 更新Top5准确率统计

            # =============================== 时间统计和日志输出 ===============================
            # 更新批次处理时间统计
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # 按照指定频率打印训练信息
            # args.print_freq控制打印频率，例如每80个batch打印一次
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, top1=top1_m, top5=top5_m))
                # 说明：val是当次传进去的值，avg是整体平均值
        
        # 返回训练统计结果
        return loss_m, top1_m, conf_mat

    @staticmethod
    def evaluate(data_loader, model, loss_f, device, classes):
        """
        评估模型性能
        
        在验证集或测试集上评估模型，不进行参数更新，只计算性能指标
        
        @param data_loader: DataLoader, 验证/测试数据加载器
        @param model: nn.Module, 要评估的神经网络模型
        @param loss_f: nn.Module, 损失函数
        @param device: torch.device, 计算设备（GPU或CPU）
        @param classes: list, 类别名称列表
        
        @return: tuple, 返回(损失统计器, Top1准确率统计器, 混淆矩阵)
        """
        # 设置模型为评估模式（禁用dropout、使用训练好的batch normalization统计）
        model.eval()

        # 获取类别数量
        class_num = len(classes)
        
        # 初始化混淆矩阵
        conf_mat = np.zeros((class_num, class_num))

        # 创建性能指标统计器
        loss_m = AverageMeter()    # 损失值统计器
        top1_m = AverageMeter()    # Top1准确率统计器
        top5_m = AverageMeter()    # Top5准确率统计器

        # 遍历验证/测试数据的所有批次
        for i, data in enumerate(data_loader):
            # 解包数据
            inputs, labels = data
            
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播（不计算梯度，节省内存和计算时间）
            outputs = model(inputs)
            
            # 计算损失
            loss = loss_f(outputs.cpu(), labels.cpu())

            # =============================== 性能计算 ===============================
            # 计算Top1和Top5准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            # 更新混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实类别索引
                pre_i = predicted[j].cpu().numpy()  # 预测类别索引
                conf_mat[cate_i, pre_i] += 1.       # 在对应位置加1

            # =============================== 指标记录 ===============================
            # 更新各种统计指标
            loss_m.update(loss.item(), inputs.size(0))      # 更新损失统计
            top1_m.update(acc1.item(), outputs.size(0))    # 更新Top1准确率统计
            top5_m.update(acc5.item(), outputs.size(0))    # 更新Top5准确率统计

        # 返回评估统计结果
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
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
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