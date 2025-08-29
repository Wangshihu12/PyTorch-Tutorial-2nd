# -*- coding:utf-8 -*-
"""
@file name  : resnet_ptq.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-25
@brief      : 肺炎Xray图像分类模型，resnet50 PTQ 量化
评估未量化前精度：
python resnet_ptq.py --mode evaluate
执行PTQ量化，并保存模型
python resnet_ptq.py --mode quantize --ptq-method max --num-data 512
python resnet_ptq.py --mode quantize --ptq-method entropy --num-data 512
python resnet_ptq.py --mode quantize --ptq-method mse --num-data 512
python resnet_ptq.py --mode quantize --ptq-method percentile --num-data 512

支持4种方法：max entropy mse percentile
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm


import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib

# 设置matplotlib后端为Agg（非交互式），适用于无图形界面的Linux服务器环境
matplotlib.use('Agg')

import utils.my_utils as utils
from datasets.pneumonia_dataset import PneumoniaDataset


def collect_stats(model, data_loader, num_batches):
    """
    前向传播，获得统计数据，并进行量化校准
    
    @param model: 要量化的模型
    @param data_loader: 数据加载器，用于提供校准数据
    @param num_batches: 用于校准的批次数量
    @return: 无返回值，直接修改模型内部状态
    """
    # 启用校准器：遍历模型中的所有模块，找到量化模块并设置校准模式
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # 如果有校准器，禁用量化，启用校准
                module.disable_quant()
                module.enable_calib()
            else:
                # 如果没有校准器，禁用该模块
                module.disable()

    # 向网络输入数据以收集统计信息：使用tqdm显示进度条
    for i, (image, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # 将图像数据移动到CUDA设备并进行前向推理
        model(image.cuda())
        # 达到指定批次数量后停止
        if i >= num_batches:
            break

    # 禁用校准器：校准完成后，重新启用量化，禁用校准
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # 启用量化，禁用校准
                module.enable_quant()
                module.disable_calib()
            else:
                # 重新启用模块
                module.enable()


def compute_amax(model, **kwargs):
    """
    根据统计值，计算amax（最大绝对值），确定量化范围的上限、下限
    用于后续计算scale（缩放因子）和Z值（零点偏移）
    
    @param model: 要量化的模型
    @param kwargs: 额外的参数，如percentile方法需要percentile值
    @return: 无返回值，直接修改模型内部状态
    """
    # 加载校准结果：遍历模型中的所有量化模块
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # 如果是MaxCalibrator，直接加载校准结果
                    module.load_calib_amax()
                else:
                    # 其他校准器，传入额外参数
                    module.load_calib_amax(**kwargs)
    
    # 将模型移动到CUDA设备
    model.cuda()


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器，定义PTQ量化所需的所有参数
    
    @param add_help: 是否添加帮助信息
    @return: 配置好的参数解析器
    """
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    
    # 数据集相关参数
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\chest_xray", type=str, help="dataset path")
    
    # 模型相关参数
    parser.add_argument("--ckpt-path", default=r"./Result/2023-09-26_01-47-40/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    # 运行模式参数
    parser.add_argument("--mode", default="quantize", type=str, help="quantize\\evaluate\\onnxexport")
    parser.add_argument("--num-data", default=512, type=int, help="量化校准数据batch数量")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    
    # PTQ量化方法参数
    parser.add_argument("--ptq-method", type=str, help="method for ptq; max; mse; entropy; percentile")

    return parser


def get_dataloader(args):
    """
    获取数据加载器，用于模型量化和评估
    
    @param args: 命令行参数对象
    @return: 训练集和验证集的数据加载器
    """
    data_dir = args.data_path
    
    # 设置图像归一化参数：均值和标准差都为0.5，适用于灰度图像
    normMean = [0.5]
    normStd = [0.5]
    # 设置模型输入图像尺寸为224x224像素
    input_size = (224, 224)
    # 创建归一化变换器
    normTransform = transforms.Normalize(normMean, normStd)
    
    # 训练集数据预处理：调整尺寸 -> 随机裁剪 -> 转换为张量 -> 归一化
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

    # 构建数据加载器：批次大小为8，工作进程数为2
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=2)
    return train_loader, valid_loader


def get_model(args, logger, device):
    """
    获取并配置模型，包括模型架构选择和权重加载
    
    @param args: 命令行参数对象
    @param logger: 日志记录器
    @param device: 计算设备（GPU或CPU）
    @return: 配置好的模型
    """
    # 根据参数选择模型架构
    if args.model == 'resnet50':
        # 创建ResNet50模型（不加载预训练权重）
        model = torchvision.models.resnet50()
    elif args.model == 'convnext':
        # 创建ConvNeXt Base模型
        model = torchvision.models.convnext_base()
    elif args.model == 'convnext-tiny':
        # 创建ConvNeXt Tiny模型
        model = torchvision.models.convnext_tiny()
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

    # ------------------------- 加载训练权重 -------------------------
    # 从检查点文件加载预训练权重
    state_dict = torch.load(args.ckpt_path)
    # 从权重字典中提取模型状态字典
    model_sate_dict = state_dict['model_state_dict']
    # 将预训练权重加载到模型中
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    # 将模型移动到指定设备
    model.to(device)
    return model


def ptq(args):
    """
    进行PTQ（Post-Training Quantization）量化，并且保存模型
    
    @param args: 命令行参数对象
    @return: 无返回值，直接保存量化后的模型
    """
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------ step1: 数据集准备 ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    
    # ------------------------------------ step2: 模型准备 ------------------------------------
    model = get_model(args, logger, device)
    
    # ------------------------------------ step3: 前向推理校准、量化 ------------------------------------
    # 使用torch.no_grad()禁用梯度计算，提高推理效率
    with torch.no_grad():
        # 设置量化模块开关，并推理，同时统计激活值分布
        collect_stats(model, train_loader, num_batches=args.num_data)

        # 根据量化方法计算amax值，确定量化范围
        if args.ptq_method == 'percentile':
            # percentile方法需要额外的percentile参数
            compute_amax(model, method='percentile', percentile=99.9)  # 计算上限、下限，并计算scale、Z值
        else:
            # 其他方法直接传入方法名
            compute_amax(model, method=args.ptq_method)                     # 计算上限、下限，并计算scale、Z值
        
        logger.info('PTQ 量化完成')
    
    # ------------------------------------ step4: 评估量化后精度 ------------------------------------
    # 定义类别标签
    classes = ["NORMAL", "PNEUMONIA"]
    # 选择交叉熵损失函数
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    
    # 在验证集上评估量化后模型的性能
    loss_m_valid, acc_m_valid, mat_valid = utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)
    logger.info('PTQ量化后模型ACC :{}，scale值计算方法是:{}'.format(acc_m_valid.avg, args.ptq_method))
    
    # ------------------------------------ step5: 保存ptq量化后模型 ------------------------------------
    # 获取检查点文件所在目录
    dir_name = os.path.dirname(args.ckpt_path)
    # 构建量化后模型的保存路径
    ptq_ckpt_path = os.path.join(dir_name, "resnet50_ptq.pth")
    # 保存量化后的模型权重
    torch.save(model.state_dict(), ptq_ckpt_path)

    # 导出ONNX格式的量化模型
    # 设置TensorQuantizer使用fake quantization模式，便于ONNX导出
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    # 为不同批次大小导出ONNX模型
    for bs in [1, 32]:
        # 构建ONNX文件名，包含批次大小、校准数据数量、量化方法和准确率信息
        model_name = "resnet_50_ptq_bs{}_data-num{}_{}_{:.2%}.onnx".format(bs, args.num_data, args.ptq_method, acc_m_valid.avg / 100)
        # 构建完整的ONNX文件路径
        onnx_path = os.path.join(dir_name, model_name)
        # 创建虚拟输入数据
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        # 导出ONNX模型
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])


def evaluate(args):
    """
    评估量化前模型的精度，用于对比量化前后的性能变化
    
    @param args: 命令行参数对象
    @return: 无返回值，直接输出评估结果
    """
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = args.output_dir
    
    # ------------------------------------  日志系统初始化 ------------------------------------
    # 创建日志记录器和日志目录
    logger, log_dir = utils.make_logger(result_dir)
    
    # ------------------------------------ step1: 数据集准备 ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    
    # ------------------------------------ step2: 模型准备 ------------------------------------
    model = get_model(args, logger, device)
    
    # ------------------------------------ step3: 模型评估 ------------------------------------
    # 定义类别标签
    classes = ["NORMAL", "PNEUMONIA"]
    # 选择交叉熵损失函数
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    
    # 在验证集上评估模型性能
    loss_m_valid, acc_m_valid, mat_valid =\
        utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

    # 记录量化前的模型准确率
    logger.info('PTQ量化前模型ACC :{}'.format(acc_m_valid.avg))


def pre_t_model_export(args):
    """
    导出FP32精度的ONNX模型，用于与量化后模型进行效率对比
    
    @param args: 命令行参数对象
    @return: 无返回值，直接保存ONNX模型
    """
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取模型
    model = get_model(args, logger, device)
    # 获取检查点文件所在目录
    dir_name = os.path.dirname(args.ckpt_path)

    # 为不同批次大小导出ONNX模型
    for bs in [1, 32]:
        # 构建FP32模型的ONNX文件名
        model_name = "resnet_50_fp32_bs{}.onnx".format(bs)
        # 构建完整的ONNX文件路径
        onnx_path = os.path.join(dir_name, model_name)
        # 创建虚拟输入数据
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        # 导出ONNX模型
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])
        # 打印保存完成信息
        print('模型保存完成: {}'.format(onnx_path))

def main(args):
    """
    主函数：根据运行模式执行相应的操作
    
    @param args: 命令行参数对象
    @return: 无返回值
    """
    if args.mode == 'quantize':
        # 量化模式：初始化量化模块，替换torch.nn的常用层为可量化的层
        quant_modules.initialize()  # 替换torch.nn的常用层，变为可量化的层
        # 执行PTQ量化
        ptq(args)
    elif args.mode == 'evaluate':
        # 评估模式：评估量化前模型精度
        evaluate(args)
    elif args.mode == 'onnxexport':
        # ONNX导出模式：导出FP32精度的ONNX模型
        pre_t_model_export(args)
    else:
        # 未知模式：打印错误信息
        print("args.mode is not recognize! got :{}".format(args.mode))


if __name__ == "__main__":
    # 解析命令行参数
    args = get_args_parser().parse_args()
    result_dir = args.output_dir
    # 创建日志记录器和日志目录
    logger, log_dir = utils.make_logger(result_dir)

    # 不指定某一种ptq_method，则进行四种量化方法的对比实验
    if args.ptq_method:
        # 如果指定了量化方法，直接执行
        main(args)
    else:
        # 如果没有指定量化方法，遍历所有支持的量化方法进行对比实验
        ptq_method_list = "max entropy mse percentile".split()
        for ptq_method in ptq_method_list:
            args.ptq_method = ptq_method
            main(args)


