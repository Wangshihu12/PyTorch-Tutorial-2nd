import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# 获取配置并初始化DeepSort跟踪器
cfg = get_config()  # 获取基础配置
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")  # 合并DeepSort配置文件
# 初始化DeepSort跟踪器，设置各种跟踪参数
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,  # ReID模型权重路径
                    max_dist=cfg.DEEPSORT.MAX_DIST,  # 最大距离阈值
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,  # 最小置信度
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,  # NMS最大重叠度
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,  # 最大IOU距离
                    max_age=cfg.DEEPSORT.MAX_AGE,  # 最大存活帧数
                    n_init=cfg.DEEPSORT.N_INIT,  # 初始化所需帧数
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,  # 特征库大小限制
                    use_cuda=True)  # 使用GPU加速


def draw_bboxes(image, bboxes, line_thickness):
    """
    在图像上绘制边界框和跟踪ID
    @param image: 输入图像，numpy数组，形状为(H, W, 3)
    @param bboxes: 边界框列表，每个元素为(x1, y1, x2, y2, cls_id, pos_id)的元组
    @param line_thickness: 线条粗细，如果为None则自动计算
    @return image: 绘制了边界框的图像
    """
    # 自动计算线条粗细，基于图像尺寸
    line_thickness = line_thickness or round(
        0.001 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []  # 存储检测点的坐标列表
    point_radius = 4  # 检测点半径

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)  # 边界框颜色：绿色

        # 计算撞线检测点：在边界框底部60%位置
        check_point_x = x1  # 检测点x坐标
        check_point_y = int(y1 + ((y2 - y1) * 0.6))  # 检测点y坐标

        # 绘制边界框
        c1, c2 = (x1, y1), (x2, y2)  # 左上角和右下角坐标
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # 绘制标签背景和文字
        font_thickness = max(line_thickness - 1, 1)  # 字体粗细
        # 计算文字尺寸
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 标签背景右下角坐标
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # 绘制填充的标签背景
        # 绘制标签文字：类别名和跟踪ID
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        # 绘制检测点（红色小方块）
        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])  # 左上角
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])  # 左下角
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])  # 右下角
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])  # 右上角

        ndarray_pts = np.array(list_pts, np.int32)  # 转换为numpy数组

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))  # 绘制红色填充多边形

        list_pts.clear()  # 清空点列表，为下一个目标做准备

    return image


def update(bboxes, image):
    """
    更新跟踪器状态并返回跟踪结果
    @param bboxes: 检测器输出的边界框列表，每个元素为(x1, y1, x2, y2, lbl, conf)的元组
    @param image: 当前帧图像，numpy数组，形状为(H, W, 3)
    @return bboxes2draw: 用于绘制的边界框列表，每个元素为(x1, y1, x2, y2, label, track_id)的元组
    """
    bbox_xywh = []  # 存储中心点格式的边界框
    confs = []  # 存储置信度
    bboxes2draw = []  # 存储最终用于绘制的边界框

    if len(bboxes) > 0:  # 如果有检测结果
        for x1, y1, x2, y2, lbl, conf in bboxes:
            # 将边界框转换为中心点格式：(center_x, center_y, width, height)
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),  # 中心点坐标
                x2 - x1, y2 - y1  # 宽度和高度
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        # 转换为PyTorch张量
        xywhs = torch.Tensor(bbox_xywh)  # 形状为(N, 4)
        confss = torch.Tensor(confs)  # 形状为(N,)

        # 更新DeepSort跟踪器
        outputs = deepsort.update(xywhs, confss, image)

        # 处理跟踪结果
        for x1, y1, x2, y2, track_id in list(outputs):
            # 计算边界框中心点
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            # 在原始检测结果中搜索最匹配的标签
            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)

            # 将结果添加到绘制列表
            bboxes2draw.append((x1, y1, x2, y2, label, track_id))
        pass
    pass

    return bboxes2draw


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在YOLOv5检测结果中搜索中心点最接近的标签
    @param center_x: 跟踪结果的中心点x坐标
    @param center_y: 跟踪结果的中心点y坐标
    @param bboxes_xyxy: 原始检测结果列表，每个元素为(x1, y1, x2, y2, lbl, conf)的元组
    @param max_dist_threshold: 最大距离阈值，用于判断是否匹配
    @return label: 匹配到的标签字符串，如果没有匹配则返回空字符串
    """
    label = ''  # 初始化标签为空
    # min_label = ''
    min_dist = -1.0  # 初始化最小距离为-1，表示未找到匹配

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        # 计算当前检测框的中心点
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 计算x和y方向的距离
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        # 判断是否在距离阈值范围内
        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 计算平均距离
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次找到匹配，直接赋值
                min_dist = avg_dist
                label = lbl
                pass
            else:
                # 如果已有匹配，选择距离更小的
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    label = lbl
                pass
            pass
        pass

    return label
