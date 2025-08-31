# -*- coding:utf-8 -*-
"""
@file name  : train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 脑部MRI图像分割训练脚本 - 使用PyTorch和segmentation_models_pytorch库
@description: 该脚本实现了脑部MRI图像分割模型的训练流程，支持UNet、UNet++、DeepLabV3+等模型架构
"""
import os
import time
import datetime

import torchvision
import torch
import torch.nn as nn
import albumentations as A  # 图像增强库
import matplotlib
import torch.optim as optim
import segmentation_models_pytorch as smp  # 分割模型库
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import utils.utils as utils
from datasets.brain_mri_dataset import BrainMRIDataset

import platform
# 在Linux系统上设置matplotlib后端为Agg（无图形界面），避免显示问题
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    """
    功能描述：创建并配置命令行参数解析器
    @param add_help: 是否添加帮助信息
    @return: 配置好的参数解析器对象
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    
    # 模型相关参数
    parser.add_argument("--encoder", default="resnet18", type=str, help="编码器类型，例如：resnet18, mobilenet_v2")
    parser.add_argument("--model", default="unet", type=str, help="模型架构：unet/unetpp/deeplabv3p")
    parser.add_argument("--device", default="cuda", type=str, help="设备选择（使用cuda或cpu，默认：cuda）")
    
    # 训练相关参数
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="每个GPU的批次大小，总批次大小为$NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="总训练轮数")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="数据加载工作进程数（默认：4）"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="优化器类型")
    parser.add_argument("--random-seed", default=42, type=int, help="随机种子")
    
    # 学习率相关参数
    parser.add_argument("--lr", default=0.01, type=float, help="初始学习率")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="动量参数")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="权重衰减（默认：1e-4）",
        dest="weight_decay",
    )
    parser.add_argument("--lr-step-size", default=20, type=int, help="每step-size个epoch降低学习率")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="学习率衰减因子")
    
    # 其他参数
    parser.add_argument("--print-freq", default=20, type=int, help="打印频率")
    parser.add_argument("--output-dir", default="./Result", type=str, help="输出保存路径")
    parser.add_argument("--resume", default="", type=str, help="检查点恢复路径")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="开始训练轮数")
    
    # 功能开关参数
    parser.add_argument('--autoaug', action='store_true', default=False, help='使用torchvision自动增强')
    parser.add_argument('--useplateau', action='store_true', default=False, help='使用ReduceLROnPlateau调度器')
    parser.add_argument('--lowlr', action='store_true', default=False, help='编码器学习率除以10')
    parser.add_argument('--bce', action='store_true', default=False, help='使用BCE损失函数')

    return parser


def main(args):
    """
    功能描述：主训练函数，执行完整的训练流程
    @param args: 命令行参数对象，包含所有训练配置
    @return: 无返回值
    """
    device = args.device
    result_dir = args.output_dir
    
    # ------------------------------------  日志设置  ------------------------------------
    # 创建日志记录器和TensorBoard写入器
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # ------------------------------------  步骤1：数据集准备  ------------------------------------
    # ImageNet预训练模型的标准化参数
    norm_mean = (0.485, 0.456, 0.406)  # RGB通道均值
    norm_std = (0.229, 0.224, 0.225)   # RGB通道标准差
    PATCH_SIZE = 256  # 图像patch大小
    
    # 训练数据增强变换
    train_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),  # 调整图像大小
        A.HorizontalFlip(p=0.5),                        # 水平翻转，概率0.5
        A.VerticalFlip(p=0.5),                          # 垂直翻转，概率0.5
        A.RandomRotate90(p=0.5),                        # 随机90度旋转，概率0.5
        A.Transpose(p=0.5),                             # 转置，概率0.5
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),  # 平移缩放旋转
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),  # 标准化
        ToTensorV2(),                                   # 转换为PyTorch张量
    ])

    # 验证数据变换（仅调整大小和标准化，无增强）
    valid_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),  # 调整图像大小
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),  # 标准化
        ToTensorV2(),   # 仅数据转换，不会除以255
    ])

    # 创建训练和验证数据集
    train_set = BrainMRIDataset(path_train, train_transform)
    valid_set = BrainMRIDataset(path_valid, valid_transform)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=args.workers)

    # ------------------------------------  步骤2：模型构建  ------------------------------------
    # 根据参数选择不同的分割模型架构
    if args.model == 'unet':
        model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    elif args.model == 'unetpp':
        model = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    elif args.model == 'deeplabv3p':
        model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    else:
        print('模型架构不被接受，必须是 unet, unetpp, deeplabv3p')
    model.to(device)  # 将模型移动到指定设备

    # ------------------------------------  步骤3：优化器和学习率调度器  ------------------------------------
    # 选择损失函数
    if args.bce:
        criterion = smp.losses.SoftBCEWithLogitsLoss()  # 二元交叉熵损失
    else:
        criterion = smp.losses.DiceLoss(mode='binary')  # Dice损失

    # 配置优化器（支持编码器不同学习率）
    if args.lowlr:
        # 编码器学习率降低10倍
        encoder_params_id = list(map(id, model.encoder.parameters()))  # 获取编码器参数ID
        base_params = filter(lambda p: id(p) not in encoder_params_id, model.parameters())  # 过滤出非编码器参数
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},  # 解码器参数使用正常学习率
            {'params': model.encoder.parameters(), 'lr': args.lr * 0.1}],  # 编码器参数使用较低学习率
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # 所有参数使用相同学习率
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 配置学习率调度器
    if args.useplateau:
        # 使用ReduceLROnPlateau：当验证指标停止改善时降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1, patience=10, cooldown=5, mode='max')
    else:
        # 使用StepLR：按固定步长降低学习率
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                            gamma=args.lr_gamma)
    
    # ------------------------------------  步骤4：训练循环  ------------------------------------
    best_miou, best_epoch = 0, 0  # 记录最佳mIoU和对应epoch
    logger.info(args)  # 记录训练参数
    logger.info('模型架构: {}'.format(str(model)))
    logger.info("开始训练")
    start_time = time.time()  # 记录训练开始时间
    epoch_time_m = utils.AverageMeter()  # 平均时间记录器
    end = time.time()
    
    # 主训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch
        loss_m_train, miou_m_train, acc_m_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, device, args, logger)
        
        # 验证
        loss_m_valid, miou_m_valid, acc_m_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device)

        # 更新epoch时间统计
        epoch_time_m.update(time.time() - end)
        end = time.time()

        # 获取当前学习率
        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        
        # 记录训练日志
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train miou avg:  {iou_train.avg:>7.4f}   '
            'Valid miou avg: {iou_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                iou_train=miou_m_train, iou_valid=miou_m_valid, lr=lr_current))

        # 学习率更新
        if args.useplateau:
            scheduler.step(miou_m_valid.avg)  # 基于验证mIoU更新学习率
        else:
            scheduler.step()  # 按步长更新学习率
            
        # TensorBoard记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('miou_group', {'train_miou': miou_m_train.avg,
                                              'valid_miou': miou_m_valid.avg}, epoch) 
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------  模型保存  ------------------------------------
        # 如果当前验证mIoU超过历史最佳，保存模型
        if best_miou < miou_m_valid.avg:
            best_epoch = epoch if best_miou < miou_m_valid.avg else best_epoch
            best_miou = miou_m_valid.avg if best_miou < miou_m_valid.avg else best_miou
            
            # 构建检查点字典
            checkpoint = {
                "model_state_dict": model.state_dict(),           # 模型状态
                "optimizer_state_dict": optimizer.state_dict(),   # 优化器状态
                "lr_scheduler_state_dict": scheduler.state_dict(), # 调度器状态
                "epoch": epoch,                                   # 当前epoch
                "args": args,                                     # 训练参数
                "best_miou": best_miou}                          # 最佳mIoU
            
            # 确定保存文件名
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)  # 保存检查点
            logger.info(f'保存检查点完成！最佳mIoU: {best_miou}, epoch: {epoch}')

    # 训练结束，记录总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("训练时间 {}".format(total_time_str))
    logger.info("最佳mIoU: {}, epoch: {}".format(checkpoint['best_miou'], checkpoint['epoch']))


if __name__ == "__main__":
    # 数据可从 https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation 下载，并通过 01_parse_data.py 解析
    path_train = r"data_train.csv"  # 训练数据CSV文件路径
    path_valid = r"data_val.csv"    # 验证数据CSV文件路径
    # path_train = r"data_train_split_by_img.csv"  # 按图像分割的训练数据路径
    # path_valid = r"data_val_split_by_img.csv"    # 按图像分割的验证数据路径

    # 解析命令行参数
    args = get_args_parser().parse_args()
    # 设置随机种子以确保结果可重现
    utils.setup_seed(args.random_seed)
    # 自动检测并设置设备（GPU或CPU）
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 开始训练
    main(args)





