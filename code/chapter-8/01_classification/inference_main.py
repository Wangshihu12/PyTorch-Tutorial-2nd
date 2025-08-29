# -*- coding:utf-8 -*-
"""
@file name  : inference_main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-02
@brief      : 推理脚本
"""
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform

# Linux系统下设置matplotlib后端为Agg（非交互式），避免显示问题
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器
    @param add_help: 是否添加帮助信息
    @return: 配置好的参数解析器
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    # 图像路径参数：指定要推理的图像文件路径
    parser.add_argument("--img-path", default=r"../../../data/imgs/person15_virus_46.jpeg", type=str, help="dataset path")
    # 模型权重文件路径：指定预训练模型的检查点文件
    parser.add_argument("--ckpt-path", default=r"./Result/2023-02-08_16-37-24/checkpoint_best.pth", type=str, help="ckpt path")
    # 模型类型：选择使用的预训练模型架构
    parser.add_argument("--model", default="convnext-tiny", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    # 设备选择：指定使用GPU或CPU进行推理
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    # 输出目录：指定结果保存的路径
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")

    return parser


def main(args):
    """
    主函数：执行图像分类推理的完整流程
    @param args: 命令行参数对象，包含所有配置信息
    """
    device = args.device
    path_img = args.img_path
    result_dir = args.output_dir
    
    # ------------------------------------ step1: 图像预处理 ------------------------------------
    
    # 设置图像归一化参数：均值和标准差都为0.5，适用于灰度图像
    normMean = [0.5]
    normStd = [0.5]
    # 设置模型输入图像尺寸为224x224像素
    input_size = (224, 224)
    # 创建归一化变换器
    normTransform = transforms.Normalize(normMean, normStd)

    # 构建图像预处理流水线：调整尺寸 -> 转换为张量 -> 归一化
    valid_transform = transforms.Compose([
        transforms.Resize(input_size),  # 将图像调整为指定尺寸
        transforms.ToTensor(),          # 将PIL图像转换为PyTorch张量
        normTransform                   # 应用归一化变换
    ])

    # 加载图像并转换为灰度图（L模式表示8位灰度图像）
    img = Image.open(path_img).convert('L')
    # 应用预处理变换
    img_tensor = valid_transform(img)
    # 将图像张量移动到指定设备（GPU或CPU）
    img_tensor = img_tensor.to(device)

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
        # 如果模型类型不匹配，打印错误信息
        print('unexpect model --> :{}'.format(args.model))

    # 获取模型名称，用于后续判断模型类型
    model_name = model._get_name()

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

    # 加载预训练权重文件
    state_dict = torch.load(args.ckpt_path)
    # 从权重字典中提取模型状态字典
    model_sate_dict = state_dict['model_state_dict']
    # 将预训练权重加载到模型中
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    # 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()
    
    # ------------------------------------ step3: 模型推理 ------------------------------------
    
    # 使用torch.no_grad()禁用梯度计算，提高推理效率
    with torch.no_grad():
        ss = time.time()  # 记录总体开始时间
        # 循环20次进行推理测试
        for i in range(20):
            s = time.time()  # 记录单次推理开始时间
            # 在第0维添加批次维度，将单张图像转换为批次格式
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 128  # 设置批次大小为128
            # 重复图像张量以创建大批次数据，用于测试模型吞吐量
            img_tensor_batch = img_tensor_batch.repeat(bs, 1, 1, 1)  # 128 or 100 or 1
            # 执行模型前向推理
            outputs = model(img_tensor_batch)
            # 对输出进行softmax归一化，得到概率分布
            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
            # 获取最大概率对应的类别索引
            _, predicted = torch.max(outputs_prob.data, 1)
            # 将预测结果转换为CPU上的numpy数组
            pred_idx = predicted.cpu().data.numpy()[0]
            # 计算单次推理耗时
            time_c = time.time() - s
            # 实时打印推理结果：预测类别、单批次耗时、吞吐量（帧/秒）
            print('\r', 'model predict: {},  speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(
                classes[pred_idx], time_c, 1*bs/time_c), end='')
        # 打印总体推理耗时
        print('\n', time.time()-ss)

    # ------------------------------------ step4: 结果可视化 ------------------------------------
    
    # 显示原始灰度图像
    plt.imshow(img, cmap='Greys_r')
    # 设置图像标题为预测结果
    plt.title("predict:{}".format(classes[pred_idx]))
    # 在图像上添加文本标注：显示预测类别和概率
    plt.text(50, 50, "predict: {}, probability: {:.1%}".format(
        classes[pred_idx], outputs_prob.cpu().data.numpy()[0, pred_idx]), bbox=dict(fc='yellow'))
    # 显示图像
    plt.show()


# 定义类别标签：正常和肺炎
classes = ["NORMAL", "PNEUMONIA"]

if __name__ == "__main__":
    # 解析命令行参数
    args = get_args_parser().parse_args()
    # 自动检测并设置设备：如果CUDA可用则使用GPU，否则使用CPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取GPU设备名称
    gpu_name = torch.cuda.get_device_name()
    # 打印GPU设备信息
    print('gpu name: {}'.format(gpu_name))
    # 调用主函数执行推理
    main(args)
