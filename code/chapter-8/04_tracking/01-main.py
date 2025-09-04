# -*- coding:utf-8 -*-
"""
@file name  : main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-21
@brief      : 采用yolov5，基于区域撞线机制，实现双向目标计数
"""

import numpy as np
import tracker
import cv2
import copy
from detector import Detector


class BoundaryType(object):
    """
    用于边界区域的mask像素填充，basecae：1和2, 由于用了插值，导致2的边界有一圈1，使得计数出错。
    采用了最近邻插值也会导致问题所在，为此，修改两个边界的索引像素，让它们差距大一些就好
    """
    inner = 68  # 内边界索引，用于矩阵像素赋值。从Outner-->inner，表示进入；蓝色区域
    outer = 168  # 外边界索引； 黄色区域


class CountBoundary(object):
    """
    计数边界类
    功能：管理单个检测区域（内圈或外圈），包括区域掩码创建、目标ID管理和计数统计
    """
    def __init__(self, point_set, mark_index, color, img_raw_shape, img_in_shape):
        """
        初始化计数边界
        @param point_set: 边界点集列表，[(x, y), (x1, y1), ...]，要求从左上角开始顺时针设置
        @param mark_index: 边界索引编号，用于区分不同边界（如内圈=1，外圈=2）
        @param color: 颜色列表 [b, g, r]，用于可视化显示
        @param img_raw_shape: 原始图像尺寸元组 (width, height)，用于创建掩码
        @param img_in_shape: 处理图像尺寸元组 (width, height)，用于缩放掩码
        """
        self.point_set = point_set  # 存储边界顶点坐标
        self.mark_index = mark_index  # 存储边界索引编号
        self.color = color  # 存储显示颜色
        self.img_raw_shape = img_raw_shape  # 原始图像尺寸
        self.img_in_shape = img_in_shape  # 处理图像尺寸

        # 目标ID容器，key是track_id，value是进入边界的序号
        self.id_container = dict()
        self.total_num = 0  # 总计数

        # 初始化区域掩码
        self._init_mask()

    def _init_mask(self):
        """
        初始化区域掩码
        功能：根据边界点集创建多边形掩码，并缩放到处理尺寸
        """
        # 将点集转换为numpy数组，数据类型为int32
        ndarray_pts = np.array(self.point_set, np.int32)
        # 创建原始尺寸的空白掩码，形状为(height, width)
        mask_raw_ = np.zeros((self.img_raw_shape[1], self.img_raw_shape[0]), dtype=np.uint8)
        # 在掩码上绘制填充多边形，颜色值为mark_index
        polygon_line_mask = cv2.fillPoly(mask_raw_, [ndarray_pts], color=self.mark_index)
        # 添加通道维度，形状变为(height, width, 1)
        polygon_line_mask = polygon_line_mask[:, :, np.newaxis]
        # 将掩码缩放到处理尺寸，使用最近邻插值保持像素值
        self.mask = cv2.resize(polygon_line_mask, self.img_in_shape, cv2.INTER_NEAREST)
        # 确保掩码有通道维度
        self.mask = self.mask[:, :, np.newaxis]

        # 创建用于可视化的彩色掩码
        mask_ = copy.deepcopy(self.mask)  # 深拷贝避免修改原掩码
        # 将掩码与颜色相乘，得到彩色显示图像
        self.mask_color = np.array(mask_ * self.color, np.uint8)

    def register_tracks(self, dets_id_list):
        """
        注册跟踪目标ID
        @param dets_id_list: 目标ID列表，包含要注册的track_id
        """
        for track_id in dets_id_list:
            self.add_id(track_id)  # 为每个ID调用添加方法

    def remove_tracks(self, dets_id_list):
        """
        移除跟踪目标ID
        @param dets_id_list: 目标ID列表，包含要移除的track_id
        """
        for track_id in dets_id_list:
            self.del_id(track_id)  # 为每个ID调用删除方法

    def add_id(self, id_):
        """
        添加单个目标ID到容器中
        @param id_: 要添加的目标ID（字符串类型，不是整数）
        """
        self.total_num += 1  # 增加总计数
        self.id_container[id_] = self.total_num  # 将ID添加到容器，值为当前计数

    def del_id(self, id_):
        """
        从容器中删除单个目标ID
        @param id_: 要删除的目标ID
        """
        self.id_container.pop(id_)  # 从字典中移除指定ID


class BaseCounter(object):
    """
    基础计数器类
    功能：实现车辆撞线计数，通过内外两个检测区域来统计车辆进出数量
    """
    def __init__(self, point_set, img_raw_shape, img_in_shape):
        """
        初始化计数器
        @param point_set: 检测线坐标点列表，包含内圈和外圈两个区域的顶点坐标
        @param img_raw_shape: 原始图像尺寸元组 (width, height)
        @param img_in_shape: 处理图像尺寸元组 (width, height)
        """
        # 创建内圈边界检测器，蓝色[255, 0, 0]
        self.inner_boundary = CountBoundary(point_set[0], BoundaryType.inner, [255, 0, 0], img_raw_shape, img_in_shape)
        # 创建外圈边界检测器，青色[0, 255, 255]
        self.outer_boundary = CountBoundary(point_set[1], BoundaryType.outer, [0, 255, 255], img_raw_shape, img_in_shape)

        # 合并两个区域的掩码，用于判断目标位置
        self.area_mask = self.inner_boundary.mask + self.outer_boundary.mask
        # 合并两个区域的颜色掩码，用于可视化显示
        self.color_img = self.inner_boundary.mask_color + self.outer_boundary.mask_color

        # 初始化计数器
        self.inner_total = 0  # 内圈计数（进入车辆数）
        self.outer_total = 0  # 外圈计数（离开车辆数）

    def counting(self, tracks):
        """
        执行撞线计数逻辑
        @param tracks: 跟踪结果列表，每个元素为(x1, y1, x2, y2, label, track_id)的元组
        """
        if len(tracks) == 0:  # 如果没有跟踪目标，直接返回
            return

        # 计算所有目标的中心点坐标
        index_x = [int((bbox[0]+bbox[2])/2) for bbox in tracks]  # 中心点x坐标列表
        index_y = [int((bbox[1]+bbox[3])/2) for bbox in tracks]  # 中心点y坐标列表
        index_yx = (index_y, index_x)  # 转换为numpy索引格式 (y, x)
        # 获取每个目标中心点在区域掩码中的值：0=无区域，1=内圈，2=外圈
        bbox_area_list = self.area_mask[index_yx]

        # ======================== 处理内圈区域 ====================================
        # 获取当前帧在内圈区域的目标ID列表
        inner_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.inner)
        # 获取历史帧中经过外圈区域的目标ID列表
        outer_tracks_history_ids = list(self.outer_boundary.id_container.keys())

        # 计算交集：当前在内圈且历史在外圈的目标（从外圈进入内圈）
        outer_2_inner_tracks_id = self.intersection(inner_tracks_currently_ids, outer_tracks_history_ids)
        # 计算差集：当前在内圈但历史不在外圈的目标（仅在内圈）
        only_at_inner_tracks_id = self.difference(inner_tracks_currently_ids, outer_tracks_history_ids)
        
        # 从外圈容器中删除已计数的目标ID
        self.outer_boundary.remove_tracks(outer_2_inner_tracks_id)
        # 在内圈容器中注册仅在内圈的目标ID
        self.inner_boundary.register_tracks(only_at_inner_tracks_id)

        # 如果有从外圈进入内圈的目标，增加内圈计数
        if len(outer_2_inner_tracks_id):
            self.inner_total += len(outer_2_inner_tracks_id)
            print('inner: {}， append: {}'.format(self.inner_total, outer_2_inner_tracks_id))

        # ======================== 处理外圈区域 ====================================
        # 获取当前帧在外圈区域的目标ID列表
        outer_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.outer)
        # 获取历史帧中经过内圈区域的目标ID列表
        inner_tracks_history_ids = list(self.inner_boundary.id_container.keys())

        # 计算交集：当前在外圈且历史在内圈的目标（从内圈进入外圈）
        inner_2_outer_tracks_id = self.intersection(outer_tracks_currently_ids, inner_tracks_history_ids)
        # 计算差集：当前在外圈但历史不在内圈的目标（仅在外圈）
        only_at_outer_tracks_id = self.difference(outer_tracks_currently_ids, inner_tracks_history_ids)
        
        # 从内圈容器中删除已计数的目标ID
        self.inner_boundary.remove_tracks(inner_2_outer_tracks_id)
        # 在外圈容器中注册仅在外圈的目标ID
        self.outer_boundary.register_tracks(only_at_outer_tracks_id)

        # 如果有从内圈进入外圈的目标，增加外圈计数
        if len(inner_2_outer_tracks_id):
            self.outer_total += len(inner_2_outer_tracks_id)
            print('outer: {}， append: {}'.format(self.outer_total, inner_2_outer_tracks_id))

    @staticmethod
    def get_currently_ids_by_area(tracks, bbox_area_list_, area_index):
        """
        获取指定区域内目标的跟踪ID列表
        @param tracks: 目标跟踪结果列表，每个元素为(x1, y1, x2, y2, label, track_id)的元组
        @param bbox_area_list_: 目标位置对应的区域索引列表，0=无区域，1=内圈，2=外圈
        @param area_index: 目标区域索引，BoundaryType.inner=1或BoundaryType.outer=2
        @return: 指定区域内目标的跟踪ID列表
        """
        # 找到指定区域内的目标索引
        area_bbox_index = np.argwhere(bbox_area_list_.squeeze() == area_index).squeeze()
        # 提取指定区域内的目标
        area_tracks = np.array(tracks)[area_bbox_index]
        # 如果只有一个目标，添加维度
        if len(area_tracks.shape) == 1:
            area_tracks = area_tracks[np.newaxis, :]
        # 提取跟踪ID（最后一列）
        area_tracks_currently_ids = list(area_tracks[:, -1])
        return area_tracks_currently_ids

    @staticmethod
    def intersection(aa, bb):
        """
        计算两个列表的交集
        @param aa: 第一个列表
        @param bb: 第二个列表
        @return: 交集列表
        """
        return list(set(aa).intersection(set(bb)))

    @staticmethod
    def difference(aa, bb):
        """
        计算两个列表的差集（aa中有，bb中没有的元素）
        @param aa: 第一个列表
        @param bb: 第二个列表
        @return: 差集列表
        """
        return list(set(aa).difference(set(bb)))


def main():
    """
    主函数：车辆跟踪与计数系统
    功能：读取视频文件，进行目标检测、跟踪和撞线计数，输出处理后的视频
    """
    path_video = r'G:\虎门大桥车流\DJI_0049.MP4'  # 输入视频文件路径
    # path_video = r'G:\DJI_0690.MP4'
    path_output_video = 'track_video.mp4'  # 输出视频文件路径
    path_yolov5_ckpt = r'F:\pytorch-tutorial-2nd\code\chapter-8\03_detection\yolov5-master\best.pt'  # YOLOv5模型权重路径
    
    # 定义检测线坐标点（多边形顶点）
    # outer_point_set = [(1772, 1394), (2088, 1388), (2102, 1494), (1730, 1452)]  # 外圈检测线
    # inner_point_set = [(1696, 1542), (2160, 1548), (2152, 1616), (1664, 1628)]  # 内圈检测线
    # 0048视频的检测线坐标
    # outer_point_set = [(845, 626), (1175, 630), (1175, 661), (838, 648)]
    # inner_point_set = [(822, 736), (1231, 735), (1227, 776), (796, 776)]
    # 0049视频的检测线坐标
    outer_point_set = [(616, 666), (1235, 655), (1245, 715), (600, 701)]  # 外圈检测线顶点坐标
    inner_point_set = [(560, 808), (1238, 812), (1243, 848), (556, 837)]  # 内圈检测线顶点坐标

    # 打开视频文件
    capture = cv2.VideoCapture(path_video)  # 创建视频捕获对象
    w_raw, h_raw = int(capture.get(3)), int(capture.get(4))  # 获取原始视频的宽度和高度
    raw_size_wh = (w_raw, h_raw)  # 原始尺寸元组 (width, height)
    in_size_wh = (1280, 720)  # 处理尺寸元组，统一调整为1280x720

    # 获取视频的基本信息
    fps = int(capture.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    # 创建视频写入器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码格式为MP4
    out = cv2.VideoWriter(path_output_video, fourcc, fps, in_size_wh)  # 创建视频写入器

    # 初始化计数器和检测器
    counter = BaseCounter([inner_point_set, outer_point_set], raw_size_wh, in_size_wh)  # 创建撞线计数器
    detector = Detector(path_yolov5_ckpt)  # 初始化YOLOv5目标检测器

    # 设置文字显示位置（在图像左上角1%位置）
    draw_text_postion = (int(in_size_wh[0] * 0.01), int(in_size_wh[1] * 0.05))

    # 主循环：逐帧处理视频
    while True:
        _, im = capture.read()  # 读取一帧图像，im为numpy数组，形状为(H, W, 3)
        if im is None:  # 如果读取失败（视频结束）
            break

        # 目标检测阶段
        im = cv2.resize(im, in_size_wh)  # 将图像调整为统一尺寸1280x720
        bboxes = detector.detect(im)  # 执行目标检测，返回边界框列表，每个元素为(x1,y1,x2,y2,label,conf)

        # 目标跟踪阶段
        if len(bboxes) > 0:  # 如果检测到目标
            list_bboxs = tracker.update(bboxes, im)  # 更新跟踪器，获取跟踪结果
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)  # 在图像上绘制边界框和跟踪ID
        else:  # 如果没有检测到目标
            output_image_frame = im  # 直接使用原图像
            list_bboxs = []  # 空跟踪结果列表

        # 撞线计数阶段
        counter.counting(list_bboxs)  # 执行撞线计数逻辑

        # 图像可视化处理
        text_draw = "In: {}, Out: {}".format(counter.inner_total, counter.outer_total)  # 生成计数显示文字
        output_image_frame = cv2.add(output_image_frame, counter.color_img)  # 将检测线叠加到输出图像上
        # 在图像上绘制计数文字
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,  # 文字位置
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                         fontScale=1,  # 字体大小
                                         color=(255, 255, 255),  # 白色文字
                                         thickness=2)  # 文字粗细
        
        # 输出处理结果
        out.write(output_image_frame)  # 将处理后的帧写入输出视频文件
        cv2.imshow('demo', output_image_frame)  # 显示处理后的图像
        cv2.waitKey(1)  # 等待1毫秒，允许窗口更新

    # 清理资源
    capture.release()  # 释放视频捕获对象
    out.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    print(text_draw)  # 打印最终的计数结果


if __name__ == '__main__':
    main()


