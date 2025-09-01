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
import cv2

import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from datetime import datetime
import logging
import segmentation_models_pytorch as smp


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def cv_imwrite(path_file, img):
    _ = cv2.imencode(".jpg", img)[1].tofile(path_file)
    return True


class ModelTrainer(object):
    """
    模型训练器类，用于图像分割任务的训练和评估
    提供训练一个epoch和模型评估的静态方法
    """

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, device, args, logger):
        """
        训练模型一个epoch的函数
        :param data_loader: 数据加载器，提供训练数据
        :param model: 待训练的神经网络模型
        :param loss_f: 损失函数，用于计算模型预测与真实标签的差异
        :param optimizer: 优化器，用于更新模型参数
        :param device: 设备类型(CPU或GPU)
        :param args: 训练参数配置对象，包含打印频率等设置
        :param logger: 日志记录器，用于输出训练信息
        :return: 返回损失、mIoU和准确率的平均值记录器
        """
        model.train()  # 设置模型为训练模式，启用dropout和batch normalization
        end = time.time()  # 记录开始时间，用于计算batch处理时间

        # 初始化各种指标的平均值记录器
        loss_m = AverageMeter()      # 损失值记录器
        miou_m = AverageMeter()      # 平均IoU记录器
        acc_m = AverageMeter()       # 准确率记录器
        batch_time_m = AverageMeter() # 批次处理时间记录器

        last_idx = len(data_loader) - 1  # 获取最后一个batch的索引
        
        # 遍历数据加载器中的每个batch
        for batch_idx, data in enumerate(data_loader):
            # 步骤1：数据准备
            inputs, labels = data  # 解包获取输入图像和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备

            # 步骤2：前向传播和反向传播
            outputs = model(inputs)  # 模型前向传播，获取预测结果
            optimizer.zero_grad()    # 清零梯度，避免梯度累积

            # 根据损失函数类型选择合适的标签格式
            if 'BCE' in loss_f._get_name():
                # 如果是二元交叉熵损失，将标签转换为浮点型
                loss = loss_f(outputs.squeeze(), labels.float())
            else:
                # 其他损失函数使用原始标签
                loss = loss_f(outputs.squeeze(), labels)
            
            loss.backward()      # 反向传播，计算梯度
            optimizer.step()     # 更新模型参数

            # 步骤3：计算评估指标（mIoU和准确率）
            # 将模型输出通过sigmoid函数转换为概率，然后二值化
            outputs = (outputs.sigmoid() > 0.5).float()
            labels = labels.unsqueeze(dim=1)  # 为标签添加通道维度，变为[bs, 1, h, w]
            
            # 计算混淆矩阵的各个分量：真正例(tp)、假正例(fp)、假负例(fn)、真负例(tn)
            # Shape of the mask should be [bs, num_classes, h, w] ,for binary segmentation num_classes = 1
            tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
            
            # 计算IoU分数，使用macro-imagewise减少策略
            # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

            # 步骤4：更新指标记录器
            # 因update里： self.sum += val * n， 因此需要传入batch数量进行加权平均
            loss_m.update(loss.item(), inputs.size(0))          # 更新损失记录器
            miou_m.update(iou_score.item(), outputs.size(0))    # 更新IoU记录器
            acc_m.update(acc.item(), outputs.size(0))           # 更新准确率记录器

            # 步骤5：记录和打印训练信息
            batch_time_m.update(time.time() - end)  # 更新批次处理时间
            end = time.time()  # 重置时间戳
            
            # 按照指定频率打印训练信息
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'miou: {miou.val:>7.4f} ({miou.avg:>7.4f})  '
                    'acc: {acc.val:>7.4f} ({acc.avg:>7.4f})'.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, miou=miou_m, acc=acc_m))
                        # val是当次传进去的值，avg是整体平均值
        return loss_m, miou_m, acc_m

    @staticmethod
    def evaluate(data_loader, model, loss_f, device):
        """
        模型评估函数，在验证集或测试集上评估模型性能
        :param data_loader: 数据加载器，提供评估数据
        :param model: 待评估的神经网络模型
        :param loss_f: 损失函数，用于计算验证损失
        :param device: 设备类型(CPU或GPU)
        :return: 返回损失、mIoU和准确率的平均值记录器
        """
        model.eval()  # 设置模型为评估模式，关闭dropout和batch normalization

        # 初始化各种指标的平均值记录器
        loss_m = AverageMeter()   # 损失值记录器
        miou_m = AverageMeter()   # 平均IoU记录器
        acc_m = AverageMeter()    # 准确率记录器

        # 遍历验证数据加载器中的每个batch
        for i, data in enumerate(data_loader):
            # 步骤1：数据准备
            inputs, labels = data  # 解包获取输入图像和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备

            # 步骤2：前向传播（无需计算梯度）
            outputs = model(inputs)  # 模型前向传播，获取预测结果

            # 根据损失函数类型计算损失
            if 'BCE' in loss_f._get_name():
                # 如果是二元交叉熵损失，将标签转换为浮点型
                loss = loss_f(outputs.squeeze(), labels.float())
            else:
                # 其他损失函数使用原始标签
                loss = loss_f(outputs.squeeze(), labels)

            # 步骤3：计算评估指标（mIoU和准确率）
            outputs = outputs.sigmoid()              # 通过sigmoid函数将输出转换为概率
            outputs = (outputs > 0.5).float()       # 二值化预测结果，阈值为0.5
            labels = labels.unsqueeze(dim=1)        # 为标签添加通道维度

            # 计算混淆矩阵的各个分量
            tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
            
            # 计算IoU分数和准确率，使用macro-imagewise减少策略
            # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

            # 步骤4：更新指标记录器
            # 因update里： self.sum += val * n， 因此需要传入batch数量进行加权平均
            loss_m.update(loss.item(), inputs.size(0))       # 更新损失记录器
            miou_m.update(iou_score.item(), outputs.size(0)) # 更新IoU记录器
            acc_m.update(acc.item(), outputs.size(0))        # 更新准确率记录器

        return loss_m, miou_m, acc_m



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