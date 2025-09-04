# -*- coding:utf-8 -*-
"""
@file name  : 00-draw-border.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-21
@brief      : 对视频进行选点， 双击图像，可获得坐标点的信息。 由于cv2.fillPoly函数的要求，需要从左上角，顺时针选点。
"""

import numpy as np
import cv2


def main():
    """
    主函数：用于视频边界绘制工具
    功能：打开视频文件，通过鼠标交互在视频帧上绘制多边形边界
    """
    # path_video = r'G:\虎门大桥车流\DJI_0048.MP4'
    path_video = r'G:\虎门大桥车流\DJI_0047.MP4'  # 视频文件路径
    # path_video = r'G:\DJI_0049.MP4'
    capture = cv2.VideoCapture(path_video)  # 打开视频文件，返回视频捕获对象
    scale = 0.8  # 图片缩放比例，用于调整显示尺寸，避免图片过大影响可视化效果

    _, img = capture.read()  # 读取视频的第一帧，img为numpy数组，形状为(H, W, 3)
    img = cv2.resize(img, None, fx=scale, fy=scale)  # 按scale比例缩放图像

    # 坐标转换函数：将缩放后的坐标转换回原始尺寸坐标
    def point_change(x_, y_, scale_):
        """
        坐标转换函数
        @param x_: 缩放后的x坐标
        @param y_: 缩放后的y坐标  
        @param scale_: 缩放比例
        @return: 原始尺寸下的坐标元组 (x, y)
        """
        return int(x_/scale_), int(y_/scale_)

    def mouseHandler(event, x, y, flags, param):
        """
        鼠标事件处理函数
        @param event: 鼠标事件类型（如左键双击、右键单击等）
        @param x: 鼠标当前x坐标
        @param y: 鼠标当前y坐标
        @param flags: 鼠标事件标志
        @param param: 额外参数
        """
        global imgCopy  # 声明使用全局变量imgCopy
        imgCopy = img.copy()  # 复制原图像，避免修改原始图像
        
        # 鼠标左键双击事件处理
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # 将缩放坐标转换为原始尺寸坐标
            raw_x, raw_y = point_change(x, y, scale)
            point_set.append((x, y))  # 将缩放后的坐标添加到点集
            point_set_raw.append((raw_x, raw_y))  # 将原始坐标添加到原始点集

            point_arr = np.array(point_set)  # 将点集转换为numpy数组，形状为(N, 2)
            # 在图像上绘制填充多边形，颜色为绿色[0, 255, 0]
            imgCopy = cv2.fillPoly(imgCopy, [point_arr], color=[0, 255, 0])
            cv2.imshow('win', imgCopy)  # 显示绘制后的图像

            # 打印当前点击的坐标信息
            print("点坐标x,y:{},{}, 原始尺寸坐标为:{},{}".format(x, y, raw_x, raw_y))
            print("点前点集：{}".format(point_set_raw))
        
        # 鼠标右键单击事件处理：清空所有已绘制的点
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_set.clear()  # 清空缩放坐标点集
            point_set_raw.clear()  # 清空原始坐标点集
            cv2.imshow('win', imgCopy)  # 显示清空后的图像

    # 创建名为'win'的窗口
    cv2.namedWindow('win')
    
    # 声明全局变量，用于存储鼠标点击的坐标点
    global point_set, point_set_raw
    point_set = []  # 存储缩放后的坐标点
    point_set_raw = []  # 存储原始尺寸的坐标点
    
    # 将鼠标回调函数绑定到'win'窗口
    cv2.setMouseCallback('win', mouseHandler)
    cv2.imshow('win', img)  # 显示原始图像
    cv2.waitKey()  # 等待键盘输入，程序暂停运行


if __name__ == '__main__':
    main()


