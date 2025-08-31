# -*- coding:utf-8 -*-
"""
@file name  : train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 肺炎Xray图像分类训练脚本
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
import matplotlib

# 设置matplotlib后端为Agg（非交互式），适用于无图形界面的Linux服务器环境
matplotlib.use('Agg')

import utils.my_utils as utils
from datasets.pneumonia_dataset import PneumoniaDataset


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器，定义训练所需的所有超参数
    @param add_help: 是否添加帮助信息
    @return: 配置好的参数解析器
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    # 数据集相关参数
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\chest_xray", type=str, help="dataset path")
    
    # 模型相关参数
    parser.add_argument("--model", default="convnext-tiny", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    # 训练相关参数
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    
    # 优化器相关参数
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    
    # 学习率调度器参数
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    
    # 其他训练参数
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    
    # 数据增强策略选择
    parser.add_argument('--autoaug', action='store_true', default=False, help='use torchvision autoaugment')
    parser.add_argument('--useplateau', action='store_true', default=False, help='use torchvision autoaugment')

    return parser


def main(args):
    """
    主函数：执行完整的肺炎Xray图像分类模型训练流程
    @param args: 命令行参数对象，包含所有训练配置信息
    """
    device = args.device
    data_dir = args.data_path
    result_dir = args.output_dir
    
    # ------------------------------------  日志系统初始化 ------------------------------------
    # 创建日志记录器和日志目录
    logger, log_dir = utils.make_logger(result_dir)
    # 创建TensorBoard写入器，用于记录训练过程
    writer = SummaryWriter(log_dir=log_dir)
    
    # ------------------------------------ step1: 数据集准备 ------------------------------------
    
    # 设置图像归一化参数：均值和标准差都为0.5，适用于灰度图像
    normMean = [0.5]
    normStd = [0.5]
    # 设置模型输入图像尺寸为224x224像素
    input_size = (224, 224)
    # 创建归一化变换器
    normTransform = transforms.Normalize(normMean, normStd)
    
    # 训练集数据增强：调整尺寸 -> 随机裁剪 -> 转换为张量 -> 归一化
    train_transform = transforms.Compose([
        transforms.Resize(256),                  # 将图像调整为256x256尺寸
        transforms.RandomCrop(input_size, padding=4),  # 随机裁剪为224x224，填充4像素
        transforms.ToTensor(),                   # 转换为PyTorch张量
        normTransform                           # 应用归一化变换
    ])

    # 验证集数据预处理：调整尺寸 -> 转换为张量 -> 归一化（无数据增强）
    valid_transform = transforms.Compose([
        transforms.Resize(input_size),           # 将图像调整为224x224尺寸
        transforms.ToTensor(),                   # 转换为PyTorch张量
        normTransform                           # 应用归一化变换
    ])

    # 数据集路径设置
    # chest_xray.zip 解压后，获得 chest_xray/train, chest_xray/test 目录结构
    # 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
    train_dir = os.path.join(data_dir, 'train')     # 训练集目录
    valid_dir = os.path.join(data_dir, 'test')      # 验证集目录
    
    # 创建数据集实例，应用相应的数据变换
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)

    # 构建数据加载器
    # 训练集：随机打乱，指定批次大小和工作进程数
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # 验证集：不打乱，固定批次大小为8，指定工作进程数
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=args.workers)

    # ------------------------------------ step2: 模型初始化 ------------------------------------
    
    # 根据参数选择预训练模型架构
    if args.model == 'resnet50':
        # 加载预训练的ResNet50模型
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'convnext':
        # 加载预训练的ConvNeXt Base模型
        model = torchvision.models.convnext_base(pretrained=True)
    elif args.model == 'convnext-tiny':
        # 加载预训练的ConvNeXt Tiny模型
        model = torchvision.models.convnext_tiny(pretrained=True)
    else:
        # 如果模型类型不匹配，记录错误日志
        logger.error('unexpect model --> :{}'.format(args.model))

    # 获取模型名称，用于后续判断模型类型
    model_name = model._get_name()

    # 根据模型类型进行适配：将RGB三通道输入改为灰度图单通道输入
    if 'ResNet' in model_name:
        # 替换ResNet模型的第一层卷积层：因为预训练模型输入是3通道RGB图像，而本案例是灰度图，输入是1通道
        # 参数说明：输入通道数=1（灰度图），输出通道数=64，卷积核大小=7x7，步长=2x2，填充=3x3，不使用偏置项
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 获取全连接层的输入特征维度
        num_ftrs = model.fc.in_features  # 替换最后一层
        # 替换最后的全连接层：输入特征维度保持不变，输出维度改为2（二分类任务：正常/肺炎）
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换ConvNeXt模型的第一层卷积层：同样适配灰度图像输入
        # 根据模型类型确定卷积核数量：convnext base为128，convnext-tiny为96
        num_kernel = 128 if args.model == 'convnext' else 96
        # 替换第一层卷积：输入通道数=1，输出通道数根据模型类型确定，卷积核大小=4x4，步长=4x4
        model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层分类器：输入特征维度保持不变，输出维度改为2
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    # 将模型移动到指定设备（GPU或CPU）
    model.to(device)

    # ------------------------------------ step3: 优化器和学习率调度器 ------------------------------------
    
    # 选择交叉熵损失函数，适用于多分类任务
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    
    # 选择SGD优化器，设置学习率、动量和权重衰减
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)  # 选择优化器
    
    # 根据参数选择学习率调度策略
    if args.useplateau:
        # 使用ReduceLROnPlateau调度器：当验证集性能不再提升时降低学习率
        # 参数说明：factor=0.2（学习率衰减因子），patience=10（等待10个epoch），cooldown=5（冷却期），mode='max'（监控指标最大化）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.2, patience=10, cooldown=5, mode='max')
    else:
        # 使用StepLR调度器：每隔固定epoch数降低学习率
        # 参数说明：step_size=20（每20个epoch降低一次），gamma=0.1（学习率衰减因子）
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                            gamma=args.lr_gamma)  # 设置学习率下降策略

    # ------------------------------------ step4: 训练迭代 ------------------------------------
    
    # 初始化最佳准确率和对应epoch
    best_acc, best_epoch = 0, 0
    
    # 记录训练参数和开始信息
    logger.info(args)
    # logger.info(train_loader, valid_loader)
    logger.info("开始训练")
    
    # 记录训练开始时间和epoch时间统计
    start_time = time.time()
    epoch_time_m = utils.AverageMeter()  # 创建epoch时间统计器
    end = time.time()
    
    # 主训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch：返回训练损失、准确率和混淆矩阵
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        
        # 验证：返回验证损失、准确率和混淆矩阵
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        # 更新epoch时间统计
        epoch_time_m.update(time.time() - end)
        end = time.time()

        # 获取当前学习率：根据调度器类型选择不同的获取方式
        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        
        # 记录详细的训练日志：包含epoch进度、时间统计、损失值、准确率和学习率信息
        logger.info(
            '当前epoch/总epoch数: [{:0>3}/{:0>3}]  '  # 显示当前epoch/总epoch数，格式化为3位数字（如：001/100）
            '当前epoch耗时(平均耗时): {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '  # 当前epoch耗时(平均耗时)，保留3位小数
            '训练集平均损失: {loss_train.avg:>6.4f}  '  # 训练集平均损失，右对齐6位，保留4位小数
            '验证集平均损失: {loss_valid.avg:>6.4f}  '  # 验证集平均损失，右对齐6位，保留4位小数
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '  # 训练集Top-1准确率，右对齐7位，保留4位小数
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '  # 验证集Top-1准确率，右对齐7位，保留4位小数
            '当前学习率: {lr}'.format(  # 当前学习率
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=lr_current))

        # 学习率更新：根据调度器类型选择不同的更新方式
        if args.useplateau:
            # ReduceLROnPlateau：根据验证集准确率更新学习率
            scheduler.step(acc_m_valid.avg)
        else:
            # StepLR：按固定步长更新学习率
            scheduler.step()
        
        # ------------------------------------ TensorBoard记录 ------------------------------------
        
        # 记录训练和验证损失
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        # 记录训练和验证准确率
        writer.add_scalars('Accuracy_group', {'train_acc': acc_m_train.avg,
                                              'valid_acc': acc_m_valid.avg}, epoch)
        
        # 生成并记录训练集混淆矩阵
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        # 生成并记录验证集混淆矩阵
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        
        # 将混淆矩阵添加到TensorBoard
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        # 记录当前学习率
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        
        # 保存条件：验证集准确率提升或训练到最后一个epoch
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            # 更新最佳准确率和对应epoch
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            
            # 构建检查点字典：包含模型状态、优化器状态、调度器状态、训练信息等
            checkpoint = {
                "model_state_dict": model.state_dict(),           # 模型参数
                "optimizer_state_dict": optimizer.state_dict(),   # 优化器状态
                "lr_scheduler_state_dict": scheduler.state_dict(), # 学习率调度器状态
                "epoch": epoch,                                  # 当前epoch
                "args": args,                                    # 训练参数
                "best_acc": best_acc}                            # 最佳准确率
            
            # 根据保存条件选择文件名：最佳模型或最终模型
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            # 构建完整的检查点文件路径
            path_checkpoint = os.path.join(log_dir, pkl_name)
            # 保存检查点文件
            torch.save(checkpoint, path_checkpoint)
            # 记录保存完成信息
            logger.info(f'save ckpt done! best acc:{best_acc}, epoch:{epoch}')

    # 计算并记录总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


# 定义类别标签：正常和肺炎
classes = ["NORMAL", "PNEUMONIA"]


if __name__ == "__main__":
    # 解析命令行参数
    args = get_args_parser().parse_args()
    # 设置随机种子，确保实验可重复性
    utils.setup_seed(args.random_seed)
    # 自动检测并设置设备：如果CUDA可用则使用GPU，否则使用CPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 调用主函数执行训练
    main(args)
