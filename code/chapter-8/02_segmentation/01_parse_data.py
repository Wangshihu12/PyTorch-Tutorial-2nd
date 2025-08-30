# -*- coding:utf-8 -*-
"""
@file name  : 01_parse_data.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-04
@brief      : 解析数据，保存为dataframe形式，并划分数据集
@reference  : https://www.kaggle.com/code/truthisneverlinear/tumor-segmentation-91-accuracy-pytorch
"""
import os
import pandas as pd
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid
from sklearn.model_selection import train_test_split


def cv_imread(path_file):
    """
    使用OpenCV读取图片，支持中文路径
    
    问题背景：
    - cv2.imread() 无法直接读取包含中文字符的路径
    - 使用np.fromfile() + cv2.imdecode() 的组合可以解决这个问题
    
    @param path_file: 图片文件的路径，支持中文路径
    @return: 读取的图片数据，格式为numpy数组
    """
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def data_parse():
    """
    根据目录结构，读取图片、标签的路径及患者id，生成数据信息CSV文件
    
    功能说明：
    1. 遍历数据目录，收集所有图片和标签的路径信息
    2. 自动匹配图片和对应的掩码标签
    3. 判断每张图片是否包含肿瘤（正负样本标记）
    4. 将信息保存到CSV文件中
    
    @return: 无返回值，直接保存CSV文件
    """
    data = []

    # 获取图片信息，存储于dataframe
    # 遍历数据根目录下的所有子目录（每个子目录代表一个患者）
    for dir_ in os.listdir(data_dir):
        # 构建完整的目录路径
        dir_path = os.path.join(data_dir, dir_)
        # 检查是否为目录
        if os.path.isdir(dir_path):
            # 遍历该患者目录下的所有文件
            for filename in os.listdir(dir_path):
                # 构建完整的文件路径
                img_path = os.path.join(dir_path, filename)
                # 将患者ID和图片路径添加到数据列表中
                data.append([dir_, img_path])
        else:
            # 如果不是目录，打印警告信息
            print(f'This is not a dir --> {dir_path}')

    # 分别获取图片与标签的路径信息
    # 创建DataFrame，包含患者ID和图片路径
    df = pd.DataFrame(data, columns=["patient", "image_path"])
    # 过滤出非掩码文件（图片文件）
    df_imgs = df[~df["image_path"].str.contains("mask")]
    # 根据图片路径自动生成对应的掩码路径
    # 假设掩码文件命名规则：原文件名 + "_mask.tif"
    df_imgs["mask_path"] = df_imgs["image_path"].apply(lambda x: x[:-4] + "_mask.tif")

    # 最终df，包含患者id，图片路径，标签路径
    dff = df_imgs

    # 新增一列判断是否有肿瘤
    def pos_neg_diagnosis(mask_path):
        """
        判断掩码中是否包含肿瘤（正负样本判断）
        
        @param mask_path: 掩码文件路径
        @return: 1表示有肿瘤（正样本），0表示无肿瘤（负样本）
        """
        # 读取掩码文件，如果最大值大于0，说明有肿瘤区域
        return_value = 1 if np.max(cv_imread(mask_path)) > 0 else 0
        return return_value
    
    # 为每张图片添加诊断标签
    dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

    # 保存数据信息到CSV文件
    dff.to_csv(PATH_SAVE, index=False)


def data_analysis():
    """
    根据CSV文件，分析正负图片比例，分析每个患者的正负图片比例
    
    功能说明：
    1. 统计整体数据中正负样本的分布
    2. 分析每个患者的正负样本分布
    3. 生成可视化图表展示分析结果
    
    @return: 无返回值，直接显示图表
    """
    # 读取数据信息CSV文件
    dff = pd.read_csv(PATH_SAVE)

    # 绘制整体正负样本分布图
    # 统计诊断标签的频次，绘制柱状图
    ax = dff.diagnosis.value_counts().plot(kind="bar", stacked=True, figsize=(10, 6), color=["violet", "orange"])

    # 设置图表标签和标题
    ax.set_xticklabels(["Positive", "Negative"], rotation=45, fontsize=12)  # 设置x轴标签
    ax.set_yticklabels("Total Images", fontsize=12)  # 设置y轴标签
    ax.set_title("Distribution of Data Grouped by Diagnosis", fontsize=18, y=1.05)  # 设置标题

    # 在柱状图上添加数值标注
    for i, rows in enumerate(dff.diagnosis.value_counts().values):
        ax.annotate(int(rows), xy=(i, rows - 12), rotation=0, color="white", ha="center", verticalalignment='bottom',
                    fontsize=15, fontweight="bold")

    # 添加总图片数量信息
    ax.text(1.2, 2550, f"Total {len(dff)} images", size=15, color="black", ha="center", va="center",
            bbox=dict(boxstyle="round", fc=("lightblue")))
    # 设置图表背景色
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.show()
    
    # --------------------------------------- 观察正负图片数量比例 ------------------------------------------------------------
    # 按患者和诊断结果分组统计
    patients_by_diagnosis = dff.groupby(["patient", "diagnosis"])["diagnosis"].size().unstack().fillna(0)
    # 重命名列名
    patients_by_diagnosis.columns = ["Positive", "Negative"]
    # 绘制每个患者的正负样本分布图
    ax = patients_by_diagnosis.plot(kind="bar", stacked=True, figsize=(18, 10), color=["violet", "springgreen"],
                                    alpha=0.85)
    # 设置图例
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(False)  # 关闭网格
    # 设置坐标轴标签和标题
    ax.set_xlabel('Patients', fontsize=20)
    ax.set_ylabel('Total Images', fontsize=20)
    ax.set_title("Distribution of data grouped by patient and diagnosis", fontsize=25, y=1.005)
    plt.show()


def data_visual():
    """
    将图片、标签读取并可视化，展示样本数据
    
    功能说明：
    1. 随机选择5个正样本（有肿瘤的图片）
    2. 同时显示原图和对应的掩码标签
    3. 使用网格布局展示多张图片
    
    @return: 无返回值，直接显示可视化结果
    """
    # 读取数据信息CSV文件
    dff = pd.read_csv(PATH_SAVE)

    # 随机选择5个正样本（diagnosis == 1）
    sample_df = dff[dff["diagnosis"] == 1].sample(5).values

    sample_imgs = []

    # 读取选中的图片和掩码，并调整大小
    for i, data in enumerate(sample_df):
        # 读取图片并调整大小
        img = cv2.resize(cv2.imread(data[1]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        # 读取掩码并调整大小
        mask = cv2.resize(cv2.imread(data[2]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        # 将图片和掩码交替添加到列表中
        sample_imgs.extend([img, mask])

    # 水平拼接图片和掩码
    # sample_imgs[::2] 获取所有图片，sample_imgs[1::2] 获取所有掩码
    sample_img_arr = np.hstack(sample_imgs[::2])      # 水平拼接所有图片
    sample_mask_arr = np.hstack(sample_imgs[1::2])    # 水平拼接所有掩码

    # 创建可视化图表
    fig = plt.figure(figsize=(25., 25.))
    # 使用ImageGrid创建2行1列的网格布局
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    # 显示图片
    grid[0].imshow(sample_img_arr)
    grid[0].set_title("Images", fontsize=25)
    grid[0].axis("off")      # 关闭坐标轴
    grid[0].grid(False)      # 关闭网格

    # 显示掩码
    grid[1].imshow(sample_mask_arr)
    grid[1].set_title("Masks", fontsize=25, y=0.9)
    grid[1].axis("off")      # 关闭坐标轴
    grid[1].grid(False)      # 关闭网格

    plt.show()


def data_split():
    """
    将数据划分为训练集、验证集，这里以dataframe形式存储
    
    重要说明：
    - 必须按患者维度划分，不能按图片维度划分
    - 避免同一患者的图片同时出现在训练集和验证集中
    - 确保数据分布的独立性和模型的泛化能力
    
    @return: 无返回值，直接保存训练集和验证集CSV文件
    """
    # 读取数据信息CSV文件
    dff = pd.read_csv(PATH_SAVE)

    # 需要根据患者维度划分，不可通过图片维度划分，以下代码可用于常见的csv划分
    # 按患者ID分组
    grouped = dff.groupby('patient')
    # grouped = dff.groupby('image_path')  # bad method - 错误的方法，会导致数据泄露
    
    # 使用train_test_split进行划分，训练集占80%
    train_set, val_set = train_test_split(list(grouped), train_size=train_size, random_state=42)
    # 提取每个组中的DataFrame数据
    train_set, val_set = [ii[1] for ii in train_set], [ii[1] for ii in val_set]
    # 合并所有训练集和验证集数据
    train_df, val_df = pd.concat(train_set), pd.concat(val_set)

    # 保存训练集和验证集到CSV文件
    train_df.to_csv(PATH_SAVE_TRAIN, index=False)
    val_df.to_csv(PATH_SAVE_VAL, index=False)
    # 打印划分结果
    print(f"Train: {train_df.shape} \nVal: {val_df.shape}")


if __name__ == "__main__":
    """
    主程序入口：解析命令行参数并执行数据处理流程
    
    执行步骤：
    1. 解析命令行参数，获取数据路径
    2. 设置各种路径和参数
    3. 依次执行数据处理、分析、可视化和划分功能
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\brain-seg\kaggle_3m",
                        type=str, help="dataset path")
    args = parser.parse_args()

    # 设置数据目录路径
    data_dir = args.data_path  # xxx/kaggle_3m
    
    # 设置各种文件路径
    PATH_SAVE = 'data_info.csv'                    # 数据信息保存路径
    # PATH_SAVE_TRAIN = 'data_train_split_by_img.csv'  # 按图片划分的训练集（错误方法）
    # PATH_SAVE_VAL = 'data_val_split_by_img.csv'      # 按图片划分的验证集（错误方法）
    PATH_SAVE_TRAIN = 'data_train.csv'             # 按患者划分的训练集（正确方法）
    PATH_SAVE_VAL = 'data_val.csv'                 # 按患者划分的验证集（正确方法）
    
    # 设置其他参数
    IMG_SHOW_SIZE = 512  # 可视化时，图像大小
    train_size = 0.8     # 训练集划分比例，80%

    # 依次执行数据处理流程
    data_parse()      # 读取根目录下数据信息，存储为csv
    data_analysis()   # 分析数据数量、比例
    data_visual()     # 可视化原图与标签
    data_split()      # 划分训练集、验证集





