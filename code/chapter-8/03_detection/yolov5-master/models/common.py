# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_jupyter, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        """
        YOLOv5多后端推理类，支持多种深度学习框架和推理引擎
        
        该类提供了统一的接口来加载和运行不同格式的YOLOv5模型，支持以下后端：
        - PyTorch (*.pt)
        - TorchScript (*.torchscript)
        - ONNX Runtime (*.onnx)
        - ONNX OpenCV DNN (*.onnx --dnn)
        - OpenVINO (*_openvino_model)
        - CoreML (*.mlmodel)
        - TensorRT (*.engine)
        - TensorFlow SavedModel (*_saved_model)
        - TensorFlow GraphDef (*.pb)
        - TensorFlow Lite (*.tflite)
        - TensorFlow Edge TPU (*_edgetpu.tflite)
        - PaddlePaddle (*_paddle_model)
        - NVIDIA Triton Inference Server
        
        Args:
            weights: 模型权重文件路径
            device: 推理设备（CPU/GPU）
            dnn: 是否使用OpenCV DNN后端
            data: 数据集配置文件路径
            fp16: 是否使用半精度推理
            fuse: 是否融合模型层
        """
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        #   NVIDIA Triton Inference Server
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)  # 获取权重文件路径
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)  # 识别模型类型
        fp16 &= pt or jit or onnx or engine  # FP16 - 只在支持的格式下启用半精度
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH) - 识别BHWC格式的模型
        stride = 32  # default stride - 默认步长
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA - 检查CUDA可用性
        if not (pt or triton):
            w = attempt_download(w)  # download if not local - 如果不是本地文件则下载

        if pt:  # PyTorch - PyTorch模型加载
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)  # 加载PyTorch模型
            stride = max(int(model.stride.max()), 32)  # model stride - 获取模型步长
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names - 获取类别名称
            model.half() if fp16 else model.float()  # 设置精度
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half() - 显式赋值以便调用设备转换方法
        elif jit:  # TorchScript - TorchScript模型加载
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata - 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)  # 加载TorchScript模型
            model.half() if fp16 else model.float()  # 设置精度
            if extra_files['config.txt']:  # load metadata dict - 加载元数据字典
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']  # 从元数据中获取步长和类别名
        elif dnn:  # ONNX OpenCV DNN - ONNX OpenCV DNN推理
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')  # 检查OpenCV依赖
            net = cv2.dnn.readNetFromONNX(w)  # 使用OpenCV DNN加载ONNX模型
        elif onnx:  # ONNX Runtime - ONNX Runtime推理
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))  # 检查ONNX Runtime依赖
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']  # 设置执行提供者
            session = onnxruntime.InferenceSession(w, providers=providers)  # 创建ONNX Runtime会话
            output_names = [x.name for x in session.get_outputs()]  # 获取输出名称
            meta = session.get_modelmeta().custom_metadata_map  # metadata - 获取模型元数据
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])  # 从元数据中获取步长和类别名
        elif xml:  # OpenVINO - OpenVINO推理
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/ - 检查OpenVINO依赖
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()  # 创建OpenVINO核心
            if not Path(w).is_file():  # if not *.xml - 如果不是*.xml文件
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir - 从OpenVINO模型目录获取*.xml文件
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))  # 读取OpenVINO模型
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCHW'))  # 设置布局为NCHW
            batch_dim = get_batch(network)  # 获取批次维度
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()  # 获取静态批次大小
            executable_network = ie.compile_model(network, device_name='CPU')  # device_name="MYRIAD" for Intel NCS2 - 编译模型
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata - 加载元数据
        elif engine:  # TensorRT - TensorRT推理
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0 - 检查TensorRT版本
            if device.type == 'cpu':
                device = torch.device('cuda:0')  # TensorRT需要CUDA设备
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))  # 定义绑定元组
            logger = trt.Logger(trt.Logger.INFO)  # 创建TensorRT日志记录器
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())  # 反序列化CUDA引擎
            context = model.create_execution_context()  # 创建执行上下文
            bindings = OrderedDict()  # 绑定字典
            output_names = []  # 输出名称列表
            fp16 = False  # default updated below - 默认FP16标志
            dynamic = False  # 动态标志
            for i in range(model.num_bindings):  # 遍历所有绑定
                name = model.get_binding_name(i)  # 获取绑定名称
                dtype = trt.nptype(model.get_binding_dtype(i))  # 获取绑定数据类型
                if model.binding_is_input(i):  # 如果是输入绑定
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic - 动态形状
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))  # 设置绑定形状
                    if dtype == np.float16:
                        fp16 = True  # 启用FP16
                else:  # output - 输出绑定
                    output_names.append(name)  # 添加到输出名称列表
                shape = tuple(context.get_binding_shape(i))  # 获取绑定形状
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建空张量
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 创建绑定对象
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 绑定地址字典
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size - 获取批次大小
        elif coreml:  # CoreML - CoreML推理
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)  # 加载CoreML模型
        elif saved_model:  # TF SavedModel - TensorFlow SavedModel推理
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model - 假设是TF1 SavedModel
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)  # 加载SavedModel
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt - TensorFlow GraphDef推理
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):  # 包装冻结图
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=''), [])  # wrapped - 包装函数
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))  # 修剪图

            def gd_outputs(gd):  # 获取GraphDef输出
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef - 遍历节点
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))  # 返回输出节点

            gd = tf.Graph().as_graph_def()  # TF GraphDef - 创建图定义
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())  # 解析图定义
            frozen_func = wrap_frozen_graph(gd, inputs='x:0', outputs=gd_outputs(gd))  # 包装冻结图
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python - TensorFlow Lite推理
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate  # 尝试导入TFLite运行时
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,  # 使用TensorFlow的TFLite
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime - TensorFlow Edge TPU推理
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]  # 根据平台选择Edge TPU委托
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])  # 创建带Edge TPU委托的解释器
            else:  # TFLite - 普通TFLite推理
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model - 加载TFLite模型
            interpreter.allocate_tensors()  # allocate - 分配张量
            input_details = interpreter.get_input_details()  # inputs - 获取输入详情
            output_details = interpreter.get_output_details()  # outputs - 获取输出详情
            # load metadata - 加载元数据
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, 'r') as model:
                    meta_file = model.namelist()[0]  # 获取元数据文件
                    meta = ast.literal_eval(model.read(meta_file).decode('utf-8'))  # 解析元数据
                    stride, names = int(meta['stride']), meta['names']  # 获取步长和类别名
        elif tfjs:  # TF.js - TensorFlow.js推理
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')  # 暂不支持TF.js
        elif paddle:  # PaddlePaddle - PaddlePaddle推理
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')  # 检查PaddlePaddle依赖
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel - 如果不是*.pdmodel文件
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir - 从PaddlePaddle模型目录获取*.pdmodel文件
            weights = Path(w).with_suffix('.pdiparams')  # 权重文件路径
            config = pdi.Config(str(w), str(weights))  # 创建PaddlePaddle配置
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)  # 启用GPU
            predictor = pdi.create_predictor(config)  # 创建预测器
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])  # 获取输入句柄
            output_names = predictor.get_output_names()  # 获取输出名称
        elif triton:  # NVIDIA Triton Inference Server - NVIDIA Triton推理服务器
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')  # 检查Triton客户端依赖
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)  # 创建Triton远程模型
            nhwc = model.runtime.startswith('tensorflow')  # 检查是否为TensorFlow运行时
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')  # 不支持的格式

        # class names - 类别名称处理
        if 'names' not in locals():  # 如果没有获取到类别名称
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}  # 从数据文件加载或使用默认名称
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet - 如果是ImageNet数据集
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names - 使用人类可读的名称

        self.__dict__.update(locals())  # assign all variables to self - 将所有变量赋值给self

    def forward(self, im, augment=False, visualize=False):
        """
        YOLOv5多后端推理前向传播方法
        
        该方法根据不同的后端类型执行相应的推理操作，支持PyTorch、ONNX、TensorRT、
        OpenVINO、CoreML、PaddlePaddle、TensorFlow等多种推理引擎。
        
        Args:
            im: 输入图像张量，形状为(batch_size, channels, height, width)
            augment: 是否使用数据增强推理，默认为False
            visualize: 是否可视化特征图，默认为False
            
        Returns:
            推理结果，格式为PyTorch张量或张量列表
        """
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width - 获取输入张量的维度信息
        
        # 精度处理：如果模型支持FP16且输入不是FP16，则转换为FP16
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16 - 转换为半精度浮点数
        
        # 数据格式处理：如果模型使用NHWC格式，则进行维度转换
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3) - PyTorch的BCHW格式转换为numpy的BHWC格式

        # 根据不同的后端类型执行相应的推理
        if self.pt:  # PyTorch - PyTorch原生推理
            # 如果启用数据增强或可视化，则传递额外参数
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            
        elif self.jit:  # TorchScript - TorchScript推理
            y = self.model(im)  # 直接调用TorchScript模型
            
        elif self.dnn:  # ONNX OpenCV DNN - ONNX OpenCV DNN推理
            im = im.cpu().numpy()  # torch to numpy - 将PyTorch张量转换为numpy数组
            self.net.setInput(im)  # 设置输入
            y = self.net.forward()  # 执行前向传播
            
        elif self.onnx:  # ONNX Runtime - ONNX Runtime推理
            im = im.cpu().numpy()  # torch to numpy - 将PyTorch张量转换为numpy数组
            # 使用ONNX Runtime会话执行推理，输入为字典格式
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            
        elif self.xml:  # OpenVINO - OpenVINO推理
            im = im.cpu().numpy()  # FP32 - 转换为numpy数组（FP32精度）
            # 使用OpenVINO可执行网络进行推理，返回字典值列表
            y = list(self.executable_network([im]).values())
            
        elif self.engine:  # TensorRT - TensorRT推理
            # 处理动态批次大小
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')  # 获取输入绑定索引
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic - 如果是动态模型则重新设置形状
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)  # 更新绑定形状
                # 重新调整输出绑定的形状
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            
            s = self.bindings['images'].shape  # 获取模型期望的输入形状
            # 检查输入形状是否匹配
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            
            self.binding_addrs['images'] = int(im.data_ptr())  # 设置输入数据指针
            self.context.execute_v2(list(self.binding_addrs.values()))  # 执行TensorRT推理
            y = [self.bindings[x].data for x in sorted(self.output_names)]  # 获取输出数据
            
        elif self.coreml:  # CoreML - CoreML推理
            im = im.cpu().numpy()  # 转换为numpy数组
            im = Image.fromarray((im[0] * 255).astype('uint8'))  # 转换为PIL图像
            # im = im.resize((192, 320), Image.ANTIALIAS) - 可选的图像缩放
            y = self.model.predict({'image': im})  # coordinates are xywh normalized - 执行预测，坐标是归一化的xywh格式
            
            # 处理CoreML输出格式
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels - 将xywh转换为xyxy像素坐标
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)  # 获取置信度和类别
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)  # 拼接结果
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto) - 分割模型的反向输出
                
        elif self.paddle:  # PaddlePaddle - PaddlePaddle推理
            im = im.cpu().numpy().astype(np.float32)  # 转换为FP32 numpy数组
            self.input_handle.copy_from_cpu(im)  # 将输入数据复制到PaddlePaddle
            self.predictor.run()  # 执行推理
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]  # 获取输出数据
            
        elif self.triton:  # NVIDIA Triton Inference Server - NVIDIA Triton推理服务器
            y = self.model(im)  # 直接调用Triton远程模型
            
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU) - TensorFlow系列推理
            im = im.cpu().numpy()  # 转换为numpy数组
            
            if self.saved_model:  # SavedModel - TensorFlow SavedModel推理
                y = self.model(im, training=False) if self.keras else self.model(im)  # 根据是否为Keras模型选择调用方式
                
            elif self.pb:  # GraphDef - TensorFlow GraphDef推理
                y = self.frozen_func(x=self.tf.constant(im))  # 使用冻结图函数进行推理
                
            else:  # Lite or Edge TPU - TensorFlow Lite或Edge TPU推理
                input = self.input_details[0]  # 获取输入详情
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model - 检查是否为TFLite量化uint8模型
                
                if int8:  # 如果是量化模型
                    scale, zero_point = input['quantization']  # 获取量化参数
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale - 反量化处理
                
                self.interpreter.set_tensor(input['index'], im)  # 设置输入张量
                self.interpreter.invoke()  # 执行推理
                
                y = []
                for output in self.output_details:  # 处理每个输出
                    x = self.interpreter.get_tensor(output['index'])  # 获取输出张量
                    if int8:  # 如果是量化模型
                        scale, zero_point = output['quantization']  # 获取输出量化参数
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale - 重新量化处理
                    y.append(x)
            
            # 统一处理TensorFlow输出格式
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]  # 确保所有输出都是numpy数组
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels - 将归一化的xywh坐标转换为像素坐标

        # 结果格式统一：将numpy数组转换为PyTorch张量
        if isinstance(y, (list, tuple)):  # 如果结果是列表或元组
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]  # 单个结果或结果列表
        else:  # 如果结果是单个数组
            return self.from_numpy(y)  # 转换为PyTorch张量

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        模型预热函数：通过运行一次推理来预热模型
        预热可以优化GPU内存分配、CUDA内核编译等，提高后续推理速度
        
        @param imgsz: 输入图像尺寸，默认(1, 3, 640, 640)表示批次大小为1，3通道，640x640分辨率
        @return: 无返回值
        """
        # 定义需要预热的模型类型：PyTorch、JIT、ONNX、TensorRT引擎、SavedModel、Protobuf、Triton
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        
        # 判断是否需要预热：当模型类型为上述类型之一，且设备不是CPU或使用Triton推理时
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            # 创建空的输入张量用于预热
            # 根据是否使用FP16精度选择数据类型：FP16用torch.half，否则用torch.float
            # 张量形状为imgsz，设备为self.device
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入张量
            
            # 预热循环：JIT模型需要2次推理，其他模型只需要1次
            # JIT模型需要额外一次推理来优化编译后的代码
            for _ in range(2 if self.jit else 1):  # 预热循环次数
                self.forward(im)  # 执行前向推理进行预热

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display
                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
