# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """
    YOLOv5检测头模块
    功能：负责将特征图转换为检测结果，包括边界框坐标、置信度和类别预测
    """
    stride = None  # 步长，在模型构建时计算
    dynamic = False  # 是否强制重建网格
    export = False  # 导出模式标志

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """
        初始化检测头
        @param nc: 类别数量，默认80（COCO数据集）
        @param anchors: 锚框列表，形状为(nl, na*2)，nl为检测层数，na为每层锚框数
        @param ch: 输入通道数列表，对应每个检测层的输入特征图通道数
        @param inplace: 是否使用原地操作，默认True
        """
        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚框的输出数量：4个坐标 + 1个置信度 + nc个类别
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 每层的锚框数量
        # 初始化网格和锚框网格，用于坐标变换
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 网格坐标
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 锚框网格
        # 注册锚框为缓冲区，形状(nl, na, 2)，2表示宽高
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        # 输出卷积层列表，将特征图转换为检测结果
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace  # 是否使用原地操作

    def forward(self, x):
        """
        前向传播
        @param x: 输入特征图列表，每个元素形状为(bs, ch, h, w)
        @return: 训练时返回原始特征图，推理时返回检测结果
        """
        z = []  # 推理输出列表
        for i in range(self.nl):  # 遍历每个检测层
            x[i] = self.m[i](x[i])  # 通过卷积层处理特征图
            bs, _, ny, nx = x[i].shape  # 获取批次大小、通道数、高度、宽度
            # 重塑张量：从(bs, 255, 20, 20)到(bs, 3, 20, 20, 85)
            # 85 = 4(坐标) + 1(置信度) + 80(类别)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理模式
                # 如果网格形状不匹配，重新生成网格
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # 分割模式（边界框 + 掩码）
                    # 分离坐标、宽高、置信度和掩码
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    # 坐标变换：sigmoid激活后乘以2加上网格偏移，再乘以步长
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]
                    # 宽高变换：sigmoid激活后平方乘以锚框网格
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
                    # 拼接所有输出
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # 检测模式（仅边界框）
                    # 分离坐标、宽高和置信度
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # 坐标变换
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    # 宽高变换
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    # 拼接输出
                    y = torch.cat((xy, wh, conf), 4)
                # 重塑为(bs, na*nx*ny, no)格式并添加到输出列表
                z.append(y.view(bs, self.na * nx * ny, self.no))

        # 根据模式返回不同格式的结果
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        """
        生成网格坐标和锚框网格
        @param nx: 网格宽度，默认20
        @param ny: 网格高度，默认20
        @param i: 检测层索引，默认0
        @param torch_1_10: 是否使用torch 1.10+版本的meshgrid语法
        @return: 网格坐标和锚框网格
        """
        d = self.anchors[i].device  # 获取设备
        t = self.anchors[i].dtype   # 获取数据类型
        shape = 1, self.na, ny, nx, 2  # 网格形状：(1, na, ny, nx, 2)
        
        # 生成y和x坐标范围
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # 创建网格坐标，兼容不同torch版本
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)
        
        # 堆叠坐标并扩展形状，减去0.5作为网格偏移
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        # 生成锚框网格：锚框乘以步长后扩展形状
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """
    YOLOv5基础模型类，继承自nn.Module，提供模型的基础功能
    
    该类提供了YOLOv5模型的核心功能，包括前向传播、性能分析、层融合等。
    所有YOLOv5模型（检测、分割、分类）都继承自此类。
    """
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        """
        模型前向传播入口
        
        Args:
            x: 输入张量，形状为(batch_size, channels, height, width)
            profile: 是否进行性能分析，默认为False
            visualize: 是否进行特征可视化，默认为False
            
        Returns:
            模型输出结果
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        单次前向传播，遍历模型的所有层
        
        Args:
            x: 输入张量
            profile: 是否进行性能分析
            visualize: 是否进行特征可视化
            
        Returns:
            最终输出结果
        """
        y, dt = [], []  # outputs - 存储各层输出和性能数据
        for m in self.model:
            # 处理层的输入：如果当前层不是从上一层获取输入
            if m.f != -1:  # if not from previous layer
                # 从指定层获取输入，支持单个层索引或层索引列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 如果启用性能分析，记录当前层的性能数据
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run - 执行当前层的前向传播
            # 保存输出：如果当前层在保存列表中，则保存其输出
            y.append(x if m.i in self.save else None)  # save output
            # 如果启用可视化，保存当前层的特征图
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        分析单个层的性能，包括计算量和推理时间
        
        Args:
            m: 当前层模块
            x: 输入张量
            dt: 性能数据列表，用于存储时间信息
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix - 判断是否为最后一层
        # 计算FLOPs（浮点运算次数），使用thop库进行性能分析
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()  # 记录开始时间
        # 运行10次推理来测量平均时间
        for _ in range(10):
            m(x.copy() if c else x)
        # 计算平均推理时间（毫秒）
        dt.append((time_sync() - t) * 100)
        # 如果是第一层，打印表头
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        # 打印当前层的性能信息
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # 如果是最后一层，打印总体性能信息
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """
        融合模型的Conv2d和BatchNorm2d层，提高推理速度
        
        将卷积层和批归一化层融合为一个卷积层，减少计算量。
        融合后的模型推理速度更快，但无法进行训练。
        
        Returns:
            self: 融合后的模型实例
        """
        LOGGER.info('Fusing layers... ')
        # 遍历模型的所有模块
        for m in self.model.modules():
            # 检查是否为卷积层且包含批归一化层
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 融合卷积层和批归一化层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 删除批归一化层属性
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播函数为融合版本
                m.forward = m.forward_fuse  # update forward
        # 打印模型信息
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """
        打印模型信息，包括参数量、计算量等
        
        Args:
            verbose: 是否打印详细信息，默认为False
            img_size: 用于计算FLOPs的输入图像尺寸，默认为640
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """
        应用函数到模型张量，处理设备转换等操作
        
        重写nn.Module的_apply方法，确保检测层的特殊属性（如stride、grid等）
        也能正确应用设备转换函数。
        
        Args:
            fn: 要应用的函数，如to()、cpu()、cuda()、half()等
            
        Returns:
            self: 应用函数后的模型实例
        """
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect() - 获取最后一层（检测层）
        # 如果是检测层或分割层，需要特殊处理其属性
        if isinstance(m, (Detect, Segment)):
            # 应用函数到步长张量
            m.stride = fn(m.stride)
            # 应用函数到网格张量列表
            m.grid = list(map(fn, m.grid))
            # 应用函数到锚框网格张量列表
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """
    YOLOv5检测模型类，继承自BaseModel，用于目标检测任务
    
    Args:
        cfg: 配置文件路径或字典，默认为'yolov5s.yaml'
        ch: 输入通道数，默认为3（RGB图像）
        nc: 类别数量，如果提供会覆盖配置文件中的值
        anchors: 锚框配置，如果提供会覆盖配置文件中的值
    """
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # 处理配置文件：支持字典或yaml文件路径
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # 定义模型参数
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 如果提供了类别数量且与配置文件不同，则覆盖配置文件的值
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 如果提供了锚框配置，则覆盖配置文件的值
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 解析模型结构，生成模型和保存列表
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 生成默认的类别名称列表
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # 获取是否使用原地操作的配置
        self.inplace = self.yaml.get('inplace', True)

        # 构建步长和锚框
        m = self.model[-1]  # Detect() - 获取最后一个检测层
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride - 最小步长的2倍
            m.inplace = self.inplace
            # 定义前向传播函数，用于计算步长
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            # 计算每个检测层的步长
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # 检查锚框顺序
            check_anchor_order(m)
            # 根据步长调整锚框尺寸
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            # 初始化偏置项（只运行一次）
            self._initialize_biases()  # only run once

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        模型前向传播，支持普通推理和数据增强推理
        
        Args:
            x: 输入张量，形状为(batch_size, channels, height, width)
            augment: 是否使用数据增强，默认为False
            profile: 是否进行性能分析，默认为False
            visualize: 是否进行可视化，默认为False
            
        Returns:
            检测结果，包含边界框、置信度和类别信息
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        数据增强前向传播，通过多尺度、多翻转增强推理精度
        
        Args:
            x: 输入图像张量
            
        Returns:
            (增强后的检测结果, None)
        """
        img_size = x.shape[-2:]  # height, width - 获取图像的高度和宽度
        s = [1, 0.83, 0.67]  # scales - 缩放比例列表
        f = [None, 3, None]  # flips (2-ud, 3-lr) - 翻转类型列表（2表示上下翻转，3表示左右翻转）
        y = []  # outputs - 存储不同增强版本的输出
        for si, fi in zip(s, f):
            # 对图像进行缩放和翻转处理
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward - 对增强后的图像进行前向传播
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # 对预测结果进行反变换，恢复到原始图像尺寸
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 裁剪增强后的尾部，避免重复检测
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """
        对增强预测结果进行反变换，恢复到原始图像坐标系
        
        Args:
            p: 预测结果张量
            flips: 翻转类型（2为上下翻转，3为左右翻转）
            scale: 缩放比例
            img_size: 原始图像尺寸(height, width)
            
        Returns:
            反变换后的预测结果
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale - 对边界框坐标进行反缩放
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud - 对上下翻转进行反变换
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr - 对左右翻转进行反变换
        else:
            # 非原地操作模式：分别处理x、y坐标和宽高
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            # 重新拼接预测结果
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """
        裁剪YOLOv5增强推理的尾部，避免重复检测
        
        Args:
            y: 增强后的预测结果列表
            
        Returns:
            裁剪后的预测结果列表
        """
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5) - 检测层数量
        g = sum(4 ** x for x in range(nl))  # grid points - 计算网格点总数
        e = 1  # exclude layer count - 排除的层数
        # 计算需要裁剪的索引
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large - 裁剪大尺度检测结果
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small - 裁剪小尺度检测结果
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        """
        初始化检测层的偏置项，基于论文"Focal Loss for Dense Object Detection"
        
        Args:
            cf: 类别频率，用于平衡不同类别的检测概率
        """
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module - 获取检测模块
        for mi, s in zip(m.m, m.stride):  # from - 遍历每个检测层和对应的步长
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85) - 重塑偏置张量
            # 根据步长调整目标检测的偏置（基于每640像素图像8个目标的假设）
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            # 根据类别频率调整分类偏置，如果没有提供频率则使用默认值
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            # 将调整后的偏置重新设置为可训练参数
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    """
    解析YOLOv5模型的yaml配置文件，构建模型结构
    
    该函数根据yaml配置文件中的定义，动态构建YOLOv5模型的网络结构。
    支持深度倍数、宽度倍数等模型缩放参数，以及各种层类型的自动处理。
    
    Args:
        d: 模型配置字典，包含backbone、head、anchors等信息
        ch: 输入通道数列表，第一个元素为初始输入通道数
        
    Returns:
        nn.Sequential: 构建好的模型层序列
        list: 需要保存输出的层索引列表（用于跳跃连接）
    """
    # Parse a YOLOv5 model.yaml dictionary
    # 打印模型构建的表头信息
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    
    # 从配置字典中提取关键参数
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    
    # 设置默认激活函数
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    
    # 计算锚框数量和输出通道数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors - 锚框数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) - 输出通道数 = 锚框数 * (类别数 + 5)

    # 初始化列表和变量
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out - 层列表、保存列表、输出通道数
    
    # 遍历backbone和head中的所有层定义
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 将字符串模块名转换为实际的模块类
        m = eval(m) if isinstance(m, str) else m  # eval strings
        
        # 处理参数列表，将字符串参数转换为实际值
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        # 根据深度倍数调整层的重复次数
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain - 深度增益
        
        # 处理不同类型的层，计算输入输出通道数
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # 卷积类层：获取输入输出通道数
            c1, c2 = ch[f], args[0]  # 输入通道数、输出通道数
            if c2 != no:  # if not output - 如果不是输出层
                # 根据宽度倍数调整输出通道数，确保是8的倍数
                c2 = make_divisible(c2 * gw, 8)

            # 更新参数列表，将通道数作为前两个参数
            args = [c1, c2, *args[1:]]
            # 对于需要重复次数的层，将重复次数插入到参数中
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats - 重复次数
                n = 1  # 重置重复次数，因为已经在参数中指定
        elif m is nn.BatchNorm2d:
            # 批归一化层：输入通道数等于前一层的输出通道数
            args = [ch[f]]
        elif m is Concat:
            # 拼接层：输出通道数等于所有输入通道数的总和
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            # 检测层或分割层：添加输入通道列表
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors - 锚框数量
                # 为每个检测层创建锚框索引列表
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                # 分割层：调整输出通道数为8的倍数
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            # 收缩层：输出通道数 = 输入通道数 * 收缩因子的平方
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            # 扩展层：输出通道数 = 输入通道数 / 扩展因子的平方
            c2 = ch[f] // args[0] ** 2
        else:
            # 其他层：输出通道数等于输入通道数
            c2 = ch[f]

        # 创建层模块：如果重复次数大于1，则创建Sequential模块
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        
        # 获取模块类型名称，用于日志输出
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        
        # 计算模块的参数数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        
        # 为模块添加属性：索引、输入来源、类型、参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        
        # 打印当前层的信息
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        
        # 更新保存列表：将需要保存输出的层索引添加到save列表中
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        
        # 将模块添加到层列表中
        layers.append(m_)
        
        # 更新通道数列表
        if i == 0:
            ch = []  # 重置通道数列表
        ch.append(c2)  # 添加当前层的输出通道数
    
    # 返回构建好的模型和排序后的保存列表
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    """
    主函数：用于测试和性能分析YOLOv5模型
    
    该函数提供了多种功能：
    1. 模型性能分析（逐层分析和前向-反向传播分析）
    2. 模型测试（测试所有yolo*.yaml配置文件）
    3. 模型融合和摘要报告
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml - 模型配置文件路径')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs - 所有GPU的总批次大小')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu - 设备选择')
    parser.add_argument('--profile', action='store_true', help='profile model speed - 分析模型速度')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer - 逐层分析模型速度')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml - 测试所有yolo配置文件')
    
    # 解析命令行参数
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML - 检查YAML文件的有效性
    print_args(vars(opt))  # 打印解析后的参数
    device = select_device(opt.device)  # 选择设备（CPU或GPU）

    # 创建模型
    # 生成随机输入张量，用于测试和性能分析
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)  # 随机输入图像 [batch_size, 3, 640, 640]
    model = Model(opt.cfg).to(device)  # 根据配置文件创建模型并移动到指定设备

    # 根据不同的参数选项执行不同的功能
    if opt.line_profile:  # profile layer by layer - 逐层性能分析
        """
        逐层性能分析：分析每一层的计算时间和内存使用
        输出每层的详细信息，包括：
        - 层名称和类型
        - 输入输出形状
        - 计算时间
        - 参数数量
        """
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward - 前向-反向传播性能分析
        """
        前向-反向传播性能分析：分析整个模型的训练性能
        包括：
        - 前向传播时间
        - 反向传播时间
        - 内存使用情况
        - 计算量（FLOPs）
        """
        results = profile(input=im, ops=[model], n=3)  # 运行3次取平均值

    elif opt.test:  # test all models - 测试所有模型
        """
        测试所有模型：验证所有yolo*.yaml配置文件的有效性
        遍历models目录下的所有yolo配置文件，尝试创建模型
        用于确保所有配置文件都能正确构建模型
        """
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):  # 递归查找所有yolo*.yaml文件
            try:
                _ = Model(cfg)  # 尝试创建模型
            except Exception as e:
                print(f'Error in {cfg}: {e}')  # 打印错误信息

    else:  # report fused model summary - 报告融合模型摘要
        """
        默认行为：融合模型并输出摘要信息
        模型融合：将Conv+BN层融合为单个Conv层，提高推理速度
        摘要信息包括：
        - 模型结构
        - 参数数量
        - 计算量
        - 模型大小
        """
        model.fuse()  # 融合模型层
