# -*- coding:utf-8 -*-
"""
@file name  : inference_seg.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-05
@brief      : 推理脚本
"""
import os.path
import time
import cv2
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
import imageio
import torch.nn as nn
import matplotlib
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import utils.utils as utils
import platform

# 平台检测：如果是Linux系统，使用非交互式后端
# 这是因为Linux服务器通常没有图形界面，需要使用Agg后端来避免matplotlib错误
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器
    功能说明：定义推理脚本所需的所有命令行参数
    
    @param add_help: 是否添加帮助信息
    @return: 配置好的参数解析器
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    # 模型检查点路径参数
    parser.add_argument("--ckpt-path", default=r"./Result-exp1/2023-03-04_08-34-34/checkpoint_best.pth", type=str, help="ckpt path")
    # 编码器类型参数（如resnet18, mobilenet_v2等）
    parser.add_argument("--encoder", default="resnet18", type=str, help="encoder type, eg: resnet18, mobilenet_v2")
    # 设备选择参数（GPU或CPU）
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    # 输出目录参数
    parser.add_argument("--output-dir", default="./inference_result", type=str, help="path to save outputs")

    return parser


def main(args):
    """
    主推理函数
    功能说明：
    1. 加载预训练的分割模型
    2. 对验证集图像进行推理
    3. 生成分割结果并可视化
    4. 保存推理结果图像
    
    @param args: 命令行参数对象
    @return: 无返回值
    """
    # 获取设备配置
    device = args.device
    # 获取结果保存目录
    result_dir = args.output_dir
    # 读取验证集数据CSV文件
    df = pd.read_csv('data_val.csv')
    
    # ------------------------------------ step1: img preprocess ------------------------------------
    # 图像预处理配置
    
    # ImageNet预训练模型的标准化参数
    norm_mean = (0.485, 0.456, 0.406)  # RGB通道均值
    norm_std = (0.229, 0.224, 0.225)   # RGB通道标准差
    PATCH_SIZE = 256  # 图像处理尺寸

    # 定义验证/推理时的数据变换
    valid_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),  # 调整图像尺寸
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),  # 标准化处理
        ToTensorV2(),   # 仅数据转换，不会除以255
    ])

    # ------------------------------------ step2: model init ------------------------------------
    # 模型初始化
    
    # 创建U-Net分割模型
    # encoder_name: 编码器类型（如resnet18）
    # in_channels: 输入通道数（RGB图像为3）
    # classes: 输出类别数（二分类为1）
    model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
    
    # 加载预训练模型权重
    state_dict = torch.load(args.ckpt_path)  # 加载检查点文件
    model_sate_dict = state_dict['model_state_dict']  # 提取模型状态字典
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    # 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()  # 关闭dropout和batch normalization的随机性
    
    # ------------------------------------ step3: inference ------------------------------------
    # 推理阶段
    
    # 使用torch.no_grad()禁用梯度计算，节省内存和计算时间
    with torch.no_grad():
        ss = time.time()  # 记录总推理开始时间
        
        # 遍历验证集中的每张图像
        for idx in range(len(df)):
            # 读取图像路径和真实标签路径
            path_img, path_mask = df.iloc[idx, 1], df.iloc[idx, 2]
            # 读取图像和真实标签
            image = utils.cv_imread(path_img)  # 原始图像
            mask = utils.cv_imread(path_mask)  # 真实分割标签

            # 对图像进行预处理变换
            augmented = valid_transform(image=image, mask=image)
            img_tensor = augmented["image"]  # 获取处理后的图像张量
            img_tensor = img_tensor.to(device)  # 移动到指定设备

            s = time.time()  # 记录单张图像推理开始时间
            
            # 添加批次维度，将单张图像转换为批次格式
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 1  # 批次大小
            # 重复图像以模拟更大的批次（用于测试推理速度）
            img_tensor_batch = img_tensor_batch.repeat(bs, 1, 1, 1)  # 128 or 100 or 1

            # 模型推理并获取预测结果
            outputs = model(img_tensor_batch)  # 前向传播
            # 应用sigmoid激活函数并二值化（阈值0.5）
            outputs_prob = (outputs.sigmoid() > 0.5).float()
            # 转换为numpy数组并调整格式
            outputs_prob = outputs_prob.squeeze().cpu().numpy().astype('uint8')

            # 可视化处理：绘制轮廓线
            # 查找预测结果的轮廓
            output_contours, _ = cv2.findContours(outputs_prob, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # 查找真实标签的轮廓
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            # 在原始图像上绘制轮廓线
            if output_contours:
                cv2.drawContours(image, output_contours, -1, (0, 255, 0), 1)  # 绿色：模型预测
            if mask_contours:
                cv2.drawContours(image, mask_contours, -1, (0, 0, 255), 1)  # 红色：真实标签
            
            # 添加图例说明
            cv2.putText(image, "Red   - Ground Ture", (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "Green - Model Predict", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            # 保存结果图像
            utils.makedir(result_dir)  # 创建输出目录
            path_save = os.path.join(result_dir, os.path.basename(path_img))  # 构建保存路径
            utils.cv_imwrite(path_save, image)  # 保存图像

            # 计算并显示推理速度
            time_c = time.time() - s  # 单张图像推理时间
            # 实时显示推理速度（帧率）
            print('\r', 'speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(time_c, 1*bs/time_c), end='')

        # 显示总推理时间
        print('\n', time.time()-ss)


if __name__ == "__main__":
    # 程序入口点
    
    # 解析命令行参数
    args = get_args_parser().parse_args()
    # 自动检测并设置设备（优先使用GPU）
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取GPU名称并显示
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    # 执行主函数
    main(args)
