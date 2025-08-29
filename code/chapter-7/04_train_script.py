# -*- coding:utf-8 -*-
"""
@file name  : 04_train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-25
@brief      : 分类任务训练脚本 - 完整的CIFAR-10图像分类训练流程
"""
import os
import time
import datetime
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import my_utils as utils


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器
    
    定义了训练脚本所需的所有超参数和配置选项
    
    @param add_help: bool, 是否添加帮助信息，默认为True
    @return: argparse.ArgumentParser, 配置好的参数解析器
    """
    import argparse

    # 创建参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    # =============================== 数据相关参数 ===============================
    # 数据集路径
    parser.add_argument("--data-path", default=r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office", type=str, help="dataset path")
    
    # 模型名称
    parser.add_argument("--model", default="resnet8", type=str, help="model name")
    
    # 计算设备选择
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    # 批次大小（每个GPU的样本数）
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    
    # 总训练轮数
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    
    # 数据加载的工作进程数
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )

    # =============================== 优化器相关参数 ===============================
    # 优化器类型
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    
    # 随机种子，确保实验可重复性
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    
    # 初始学习率
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    
    # 动量参数（SGD优化器使用）
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    
    # 权重衰减（L2正则化）
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # =============================== 学习率调度参数 ===============================
    # 学习率衰减步长（每隔多少个epoch衰减一次）
    parser.add_argument("--lr-step-size", default=80, type=int, help="decrease lr every step-size epochs")
    
    # 学习率衰减因子（每次衰减乘以的系数）
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    # =============================== 输出和日志参数 ===============================
    # 打印频率（每隔多少个iteration打印一次）
    parser.add_argument("--print-freq", default=80, type=int, help="print frequency")
    
    # 输出目录路径
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    
    # 恢复训练的检查点路径
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    
    # 开始训练的epoch（用于恢复训练）
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    return parser


def main(args):
    """
    主训练函数
    
    执行完整的训练流程，包括数据准备、模型训练、验证和保存
    
    @param args: 解析后的命令行参数
    """
    # 提取关键参数
    device = args.device          # 计算设备
    data_dir = args.data_path     # 数据目录
    result_dir = args.output_dir  # 结果保存目录
    
    # =============================== 日志和可视化设置 ===============================
    # 创建日志记录器和日志目录
    logger, log_dir = utils.make_logger(result_dir)
    
    # 创建TensorBoard写入器，用于记录训练过程
    writer = SummaryWriter(log_dir=log_dir)
    
    # =============================== 步骤1：数据集准备 ===============================
    
    # CIFAR-10数据集的标准化参数（基于训练集计算得出）
    normMean = [0.4948052, 0.48568845, 0.44682974]    # RGB三通道的均值
    normStd = [0.24580306, 0.24236229, 0.2603115]     # RGB三通道的标准差
    
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

    # 加载CIFAR-10数据集
    # 注意：root变量下需要存放cifar-10-python.tar.gz文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, num_workers=args.workers)

    # =============================== 步骤2：模型构建 ===============================
    # 创建ResNet8模型实例
    model = utils.resnet8()
    
    # 将模型移动到指定设备（GPU或CPU）
    model.to(device)

    # =============================== 步骤3：损失函数、优化器和学习率调度器 ===============================
    # 选择交叉熵损失函数，适用于多分类问题
    criterion = nn.CrossEntropyLoss()
    
    # 选择SGD优化器，设置学习率、动量和权重衰减
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 设置学习率调度器：每隔lr_step_size个epoch将学习率乘以lr_gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # =============================== 步骤4：训练循环 ===============================
    # 初始化最佳性能记录
    best_acc, best_epoch = 0, 0
    
    # 记录训练开始信息
    logger.info(args)
    logger.info(train_loader, valid_loader)
    logger.info("Start training")
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 创建epoch时间记录器
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    
    # 开始训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # =============================== 训练阶段 ===============================
        # 训练一个epoch，返回训练损失、准确率和混淆矩阵
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        
        # =============================== 验证阶段 ===============================
        # 在验证集上评估模型性能
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        # 更新epoch时间统计
        epoch_time_m.update(time.time() - end)
        end = time.time()

        # 记录详细的训练信息
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=scheduler.get_last_lr()[0]))

        # 更新学习率
        scheduler.step()
        
        # =============================== TensorBoard记录 ===============================
        # 记录训练和验证损失
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        
        # 记录训练和验证准确率
        writer.add_scalars('Accuracy_group', {'train_acc': acc_m_train.avg,
                                              'valid_acc': acc_m_valid.avg}, epoch)
        
        # 生成并记录混淆矩阵图
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        
        # 将混淆矩阵添加到TensorBoard
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        
        # 记录当前学习率
        writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

        # =============================== 模型保存 ===============================
        # 保存条件：验证准确率提升或训练结束
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            # 更新最佳性能记录
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            
            # 创建检查点字典，包含所有必要的状态信息
            checkpoint = {
                "model_state_dict": model.state_dict(),           # 模型参数
                "optimizer_state_dict": optimizer.state_dict(),   # 优化器状态
                "lr_scheduler_state_dict": scheduler.state_dict(), # 学习率调度器状态
                "epoch": epoch,                                  # 当前epoch
                "args": args,                                    # 训练参数
                "best_acc": best_acc                             # 最佳准确率
            }
            
            # 确定保存文件名
            # 如果是最后一个epoch，保存为checkpoint_{epoch}.pth
            # 如果是新的最佳模型，保存为checkpoint_best.pth
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            
            # 构建完整的保存路径
            path_checkpoint = os.path.join(log_dir, pkl_name)
            
            # 保存检查点
            torch.save(checkpoint, path_checkpoint)

    # =============================== 训练完成统计 ===============================
    # 计算总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    # 记录总训练时间
    logger.info("Training time {}".format(total_time_str))


# =============================== 类别定义 ===============================
# CIFAR-10数据集的10个类别
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    # =============================== 主程序入口 ===============================
    # 解析命令行参数
    args = get_args_parser().parse_args()
    
    # 设置随机种子，确保实验可重复性
    utils.setup_seed(args.random_seed)
    
    # 自动检测并设置计算设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 开始主训练流程
    main(args)





