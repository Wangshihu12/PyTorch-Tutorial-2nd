# -*- coding:utf-8 -*-
"""
@file name  : inference_main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-02
@brief      : 推理脚本
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir):
    """
    将VisDrone数据集的标注格式转换为YOLO格式
    功能说明：
    1. 读取VisDrone格式的标注文件（.txt格式）
    2. 将边界框坐标从VisDrone格式转换为YOLO格式
    3. 生成YOLO格式的标签文件
    4. 处理多个数据集（训练集、验证集、测试集）
    
    @param dir: Path对象，指向包含images和annotations文件夹的数据集目录
    @return: 无返回值，直接生成转换后的标签文件
    """
    
    def convert_box(size, box):
        """
        将VisDrone格式的边界框转换为YOLO格式
        VisDrone格式：x1, y1, width, height（左上角坐标 + 宽高）
        YOLO格式：x_center, y_center, width, height（中心点坐标 + 宽高，归一化到0-1）
        
        @param size: 图像尺寸元组 (width, height)
        @param box: VisDrone格式的边界框元组 (x1, y1, width, height)
        @return: YOLO格式的边界框元组 (x_center, y_center, width, height)
        """
        # 计算归一化因子：将像素坐标转换为0-1范围
        dw = 1. / size[0]  # 宽度归一化因子
        dh = 1. / size[1]  # 高度归一化因子
        
        # 转换坐标格式：
        # x_center = (x1 + width/2) * dw  # 计算中心点x坐标并归一化
        # y_center = (y1 + height/2) * dh  # 计算中心点y坐标并归一化
        # width = width * dw  # 宽度归一化
        # height = height * dh  # 高度归一化
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    # 创建labels目录用于存放转换后的YOLO标签文件
    # parents=True: 如果父目录不存在则创建
    # exist_ok=True: 如果目录已存在则不报错
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory  pathlib库的作用！
    
    # 使用tqdm创建进度条，遍历annotations目录下的所有.txt文件
    # glob('*.txt'): 匹配所有.txt文件
    # desc: 进度条描述信息
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    
    # 遍历每个标注文件
    for f in pbar:
        # 获取对应的图像文件路径并读取图像尺寸
        # f.name: 获取文件名（不含路径）
        # with_suffix('.jpg'): 将文件扩展名改为.jpg
        # Image.open().size: 获取图像尺寸 (width, height)
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        
        # 存储转换后的YOLO格式标签行
        lines = []
        
        # 读取VisDrone格式的标注文件
        with open(f, 'r') as file:  # read annotation.txt
            # 逐行处理标注数据
            # file.read().strip(): 读取文件内容并去除首尾空白
            # splitlines(): 按行分割
            # x.split(','): 每行按逗号分割
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                # VisDrone数据集中，类别0表示"忽略区域"，需要跳过
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                
                # 获取目标类别ID，VisDrone从1开始，YOLO从0开始，所以需要减1
                cls = int(row[5]) - 1
                
                # 转换边界框格式
                # row[:4]: 取前4个元素作为边界框坐标 (x1, y1, width, height)
                # tuple(map(int, row[:4])): 将字符串转换为整数元组
                # convert_box(): 转换为YOLO格式
                box = convert_box(img_size, tuple(map(int, row[:4])))
                
                # 构建YOLO格式的标签行：类别ID + 边界框坐标（空格分隔）
                # f'{x:.6f}': 将浮点数格式化为6位小数
                # ' '.join(): 用空格连接所有坐标值
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                
                # 构建输出文件路径
                # 将annotations路径替换为labels路径
                # os.sep: 操作系统路径分隔符（Windows为\，Linux为/）
                txt_path = str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep)
                
                # 写入YOLO格式的标签文件
                with open(txt_path, 'w') as fl:
                    fl.writelines(lines)  # write label.txt


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(add_help=True)
    # 添加数据集路径参数，默认路径为VisDrone数据集位置
    parser.add_argument("--data-path", default=r'G:\deep_learning_data\VisDrone',
                        type=str, help="dataset path")
    # 解析命令行参数
    args = parser.parse_args()
    # 获取数据集根目录
    root_dir = args.data_path  # dataset root dir

    # 定义需要转换的数据集列表
    # VisDrone数据集包含训练集、验证集和测试集
    dataset_list = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    
    # 遍历每个数据集进行格式转换
    for name in dataset_list:
        # 调用转换函数，传入数据集路径
        # Path(root_dir, name): 构建完整的数据集路径
        visdrone2yolo(Path(root_dir, name))  # convert VisDrone annotations to YOLO labels

