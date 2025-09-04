import torch
import numpy as np

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


class Detector:
    """
    目标检测器类
    功能：基于YOLOv5模型进行目标检测，主要用于车辆和行人检测
    """

    def __init__(self, path_yolov5_ckpt):
        """
        初始化检测器
        @param path_yolov5_ckpt: YOLOv5模型权重文件路径
        """
        self.img_size = 1280  # 输入图像尺寸，YOLOv5的标准输入尺寸
        self.threshold = 0.3  # 检测置信度阈值，低于此值的检测结果将被过滤
        self.stride = 1  # 模型步长参数

        self.weights = path_yolov5_ckpt  # 存储模型权重路径

        # 设备选择：优先使用GPU，否则使用CPU
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)  # 选择计算设备
        
        # 加载YOLOv5模型
        model = attempt_load(self.weights)  # 加载模型权重
        model.to(self.device).eval()  # 将模型移至指定设备并设置为评估模式
        model.half()  # 将模型转换为半精度浮点数(FP16)，减少内存占用并提高推理速度

        self.m = model  # 存储模型实例
        # 获取类别名称列表，支持多GPU训练时的模型结构
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):
        """
        图像预处理函数
        @param img: 输入图像，numpy数组，形状为(H, W, 3)
        @return img0: 原始图像副本
        @return img: 预处理后的图像张量，形状为(1, 3, H, W)
        """
        img0 = img.copy()  # 保存原始图像副本
        
        # 图像预处理：调整尺寸、填充等
        img = letterbox(img, new_shape=self.img_size)[0]  # 将图像调整为指定尺寸，保持宽高比
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，并调整维度顺序为(3, H, W)
        img = np.ascontiguousarray(img)  # 确保数组内存连续
        img = torch.from_numpy(img).to(self.device)  # 转换为PyTorch张量并移至指定设备
        img = img.half()  # 转换为半精度浮点数
        img /= 255.0  # 归一化到[0,1]范围
        
        # 添加batch维度，如果输入是3维张量
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 添加batch维度，形状变为(1, 3, H, W)

        return img0, img

    def detect(self, im):
        """
        目标检测主函数
        @param im: 输入图像，numpy数组，形状为(H, W, 3)
        @return boxes: 检测结果列表，每个元素为(x1, y1, x2, y2, label, confidence)的元组
        """
        # 图像预处理
        im0, img = self.preprocess(im)

        # 模型推理
        pred = self.m(img, augment=False)[0]  # 执行前向推理，不使用数据增强
        pred = pred.float()  # 转换为单精度浮点数
        # 非极大值抑制(NMS)，去除重叠的检测框
        pred = non_max_suppression(pred, self.threshold, 0.4)

        boxes = []  # 存储检测结果
        for det in pred:  # 遍历每张图像的检测结果
            if det is not None and len(det):  # 如果有检测结果
                # 将检测框坐标从模型输出尺寸缩放回原始图像尺寸
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # 遍历每个检测结果
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]  # 获取类别标签
                    # 过滤特定类别的目标，只保留交通相关的目标
                    # if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                    if lbl not in ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                                   'tricycle', 'awning-tricycle', 'bus', 'motor']:
                        continue
                    
                    # 提取边界框坐标
                    x1, y1 = int(x[0]), int(x[1])  # 左上角坐标
                    x2, y2 = int(x[2]), int(x[3])  # 右下角坐标
                    
                    # 将检测结果添加到列表中：(x1, y1, x2, y2, label, confidence)
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes
