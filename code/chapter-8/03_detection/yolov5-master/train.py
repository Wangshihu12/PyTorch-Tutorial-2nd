# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

# ================================== 标准库导入 ==================================
import argparse        # 命令行参数解析库，用于处理训练脚本的命令行参数
import math           # 数学函数库，用于数学计算（如学习率调度、损失计算等）
import os             # 操作系统接口库，用于文件路径操作、环境变量读取等
import random         # 随机数生成库，用于数据打乱、随机种子设置等
import subprocess     # 子进程管理库，用于执行系统命令（如Git操作）
import sys            # 系统相关的参数和函数，用于路径管理、退出程序等
import time           # 时间相关函数，用于训练时间统计、性能监控等
from copy import deepcopy       # 深拷贝函数，用于复制模型配置、参数等
from datetime import datetime   # 日期时间处理，用于训练日志时间戳
from pathlib import Path       # 现代化路径处理库，用于跨平台文件路径操作

# ================================== 第三方库导入 ==================================
import numpy as np    # 数值计算库，用于数组操作、数学计算等
import torch          # PyTorch深度学习框架主库
import torch.distributed as dist  # PyTorch分布式训练模块，用于多GPU/多机训练
import torch.nn as nn            # PyTorch神经网络模块，包含各种网络层和损失函数
import yaml           # YAML文件解析库，用于读取配置文件
from torch.optim import lr_scheduler  # PyTorch学习率调度器，用于动态调整学习率
from tqdm import tqdm  # 进度条库，用于显示训练进度

# ================================== 路径配置 ==================================
FILE = Path(__file__).resolve()  # 获取当前脚本文件的绝对路径
ROOT = FILE.parents[0]           # 获取YOLOv5项目根目录路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   # 将YOLOv5根目录添加到Python模块搜索路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对于当前工作目录的相对路径

# ================================== YOLOv5模块导入 ==================================
# 验证和评估模块
import val as validate  # 导入验证模块，用于每个epoch结束时计算mAP等指标

# 模型相关模块
from models.experimental import attempt_load  # 实验性模型加载函数，支持多种模型格式
from models.yolo import Model                 # YOLO模型主体结构，包含网络架构定义

# 工具函数模块
from utils.autoanchor import check_anchors           # 自动锚框检查，确保锚框适合数据集
from utils.autobatch import check_train_batch_size   # 自动批次大小检查，根据GPU内存调整
from utils.callbacks import Callbacks                # 回调函数系统，用于训练过程中的事件处理
from utils.dataloaders import create_dataloader      # 数据加载器创建函数，处理数据读取和增强
from utils.downloads import attempt_download, is_url  # 下载工具，自动下载模型权重和数据集

# 通用工具函数（数量较多，包含训练所需的各种辅助功能）
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
# 主要功能说明：
# - LOGGER: 日志记录器
# - check_amp: 自动混合精度检查
# - check_dataset: 数据集完整性检查
# - init_seeds: 随机种子初始化
# - labels_to_class_weights: 根据标签计算类别权重
# - one_cycle: 单周期学习率调度
# - strip_optimizer: 优化器状态清除

# 日志记录模块
from utils.loggers import Loggers                          # 多种日志记录器集成（TensorBoard、WandB等）
from utils.loggers.comet.comet_utils import check_comet_resume  # Comet实验跟踪工具支持

# 训练核心模块
from utils.loss import ComputeLoss     # 损失函数计算，包含分类、回归、置信度损失
from utils.metrics import fitness      # 模型性能评估指标，用于模型选择
from utils.plots import plot_evolve    # 进化算法结果可视化

# PyTorch工具模块
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
# 主要功能说明：
# - EarlyStopping: 早停机制，防止过拟合
# - ModelEMA: 指数移动平均，提高模型稳定性
# - select_device: 智能设备选择（CPU/GPU）
# - smart_DDP: 智能分布式数据并行
# - smart_optimizer: 智能优化器配置

# ================================== 分布式训练环境变量 ==================================
# 获取分布式训练相关的环境变量，用于多GPU/多机训练配置
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 本地GPU排名，-1表示非分布式训练
RANK = int(os.getenv('RANK', -1))              # 全局进程排名，用于多机训练
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))   # 总进程数量，1表示单机训练
# 参考文档: https://pytorch.org/docs/stable/elastic/run.html

# ================================== 版本信息 ==================================
GIT_INFO = check_git_info()  # 获取Git仓库信息，用于版本追踪和实验记录


def train(hyp, opt, device, callbacks):
    """
    YOLOv5训练主函数
    
    @param hyp: 超参数字典或YAML文件路径，包含学习率、权重衰减等训练参数
    @param opt: 命令行选项对象，包含所有训练配置参数
    @param device: 训练设备(CPU或GPU)
    @param callbacks: 回调函数管理器，用于训练过程中的事件处理
    @return: 训练结果元组，包含精度、召回率、mAP等指标
    """
    # ================================== 第一步：参数解析和初始化 ==================================
    # 从opt对象中提取关键训练参数，简化后续代码中的参数访问
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    
    # 触发预训练准备阶段的回调函数
    callbacks.run('on_pretrain_routine_start')

    # ================================== 第二步：目录结构创建 ==================================
    # 创建权重文件保存目录
    w = save_dir / 'weights'  # weights dir - 权重文件存储目录
    # 根据是否进行进化算法来决定创建目录的层级
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir - 递归创建目录
    # 定义最新权重和最佳权重的保存路径
    last, best = w / 'last.pt', w / 'best.pt'  # 最新权重和最佳权重文件路径

    # ================================== 第三步：超参数处理 ==================================
    # 处理超参数：如果是文件路径则读取，如果是字典则直接使用
    if isinstance(hyp, str):
        # 从YAML文件加载超参数字典
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict - 安全加载YAML文件
    
    # 打印超参数信息到日志
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints - 保存超参数副本到检查点

    # ================================== 第四步：保存运行设置 ==================================
    # 如果不是进化算法模式，保存配置文件便于后续复现实验
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)      # 保存超参数配置
        yaml_save(save_dir / 'opt.yaml', vars(opt))  # 保存命令行选项配置

    # ================================== 第五步：日志记录器配置 ==================================
    data_dict = None  # 数据集字典初始化
    # 只在主进程（RANK为-1或0）中设置日志记录器，避免多进程重复日志
    if RANK in {-1, 0}:
        # 创建日志记录器实例，支持TensorBoard、WandB等多种日志系统
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # 注册日志记录器的回调函数到回调管理器中
        # 将日志记录器的所有方法注册为回调函数，实现训练过程的自动日志记录
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # 处理远程数据集链接（用于云端训练或数据集自动下载）
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact - 如果从远程恢复训练
            # 从远程恢复时，更新相关参数
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # ================================== 第六步：基础配置设置 ==================================
    plots = not evolve and not opt.noplots    # 是否生成训练可视化图表
    cuda = device.type != 'cpu'               # 检查是否使用CUDA加速
    
    # 初始化随机种子，确保训练结果的可重复性
    # 每个进程使用不同的种子，避免多进程训练时的随机性重复
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    
    # 分布式训练中确保只有第一个进程执行数据集检查，避免重复下载
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None - 检查并验证数据集
    
    # ================================== 第七步：数据集路径和类别配置 ==================================
    train_path, val_path = data_dict['train'], data_dict['val']  # 获取训练集和验证集路径
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes - 类别数量（单类检测时为1）
    # 处理类别名称：单类检测时使用默认名称，否则使用数据集定义的类别名称
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # 检查是否为COCO数据集，用于后续特殊处理
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # ================================== 第八步：模型创建和权重加载 ==================================
    check_suffix(weights, '.pt')  # 检查权重文件格式，确保是.pt格式
    pretrained = weights.endswith('.pt')  # 判断是否使用预训练权重
    
    if pretrained:
        # 使用预训练权重的模型创建流程
        with torch_distributed_zero_first(LOCAL_RANK):
            # 尝试下载权重文件（如果本地不存在）
            weights = attempt_download(weights)  # download if not found locally
        
        # 加载检查点到CPU，避免CUDA内存泄漏
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)  # load checkpoint to CPU to avoid CUDA memory leak
        
        # 创建模型：使用配置文件或检查点中的模型配置
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        
        # 决定哪些参数不加载：如果指定了新的配置或锚框，则排除原有的锚框参数
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        
        # 获取检查点中的模型状态字典，转换为FP32格式
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        
        # 计算当前模型和检查点模型的参数交集，排除指定的参数
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        
        # 加载匹配的参数到模型中，strict=False允许部分参数不匹配
        model.load_state_dict(csd, strict=False)  # load
        
        # 报告成功传输的参数数量
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 从头开始训练的模型创建流程
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    
    # 检查自动混合精度（AMP）支持
    amp = check_amp(model)  # check AMP

    # ================================== 第九步：模型层冻结设置 ==================================
    # 处理需要冻结的层：支持指定层数或具体层名称
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    
    # 遍历模型的所有参数，设置是否需要梯度更新
    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认所有层都参与训练
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        
        # 如果参数名包含在冻结列表中，则冻结该参数
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')  # 记录被冻结的参数
            v.requires_grad = False  # 冻结参数，不参与梯度更新

    # ================================== 第十步：图像尺寸配置 ==================================
    gs = max(int(model.stride.max()), 32)  # 网格大小（最大步长），用于图像尺寸对齐
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 验证并调整图像尺寸为gs的倍数

    # ================================== 第十一步：批次大小优化 ==================================
    # 在单GPU模式下自动估算最佳批次大小
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)  # 根据GPU内存自动调整批次大小
        loggers.on_params_update({'batch_size': batch_size})  # 更新日志记录器中的批次大小

    # ================================== 第十二步：优化器配置 ==================================
    nbs = 64  # nominal batch size - 标准批次大小，用于学习率和权重衰减缩放
    accumulate = max(round(nbs / batch_size), 1)  # 梯度累积步数，模拟大批次训练
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 根据有效批次大小缩放权重衰减
    # 创建智能优化器，自动选择合适的优化器类型和参数
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # ================================== 第十三步：学习率调度器配置 ==================================
    if opt.cos_lr:
        # 余弦退火学习率调度：从1衰减到hyp['lrf']
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # 线性学习率调度：线性衰减
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器

    # ================================== 第十四步：指数移动平均（EMA）配置 ==================================
    # 只在主进程中创建EMA，用于模型参数的平滑更新
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # ================================== 第十五步：训练恢复处理 ==================================
    best_fitness, start_epoch = 0.0, 0  # 初始化最佳性能指标和起始epoch
    if pretrained:
        if resume:
            # 智能恢复训练：从检查点恢复优化器、EMA、epoch等状态
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        # 清理内存，删除不再需要的检查点数据
        del ckpt, csd

    # ================================== 第十六步：多GPU训练模式配置 ==================================
    # 数据并行（DP）模式 - 不推荐使用，建议使用DDP
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)  # 使用数据并行模式

    # 同步批归一化 - 在分布式训练中同步不同GPU上的BN统计信息
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')  # 记录使用同步BN

    # ================================== 第十七步：训练数据加载器创建 ==================================
    # 创建训练数据加载器和数据集
    train_loader, dataset = create_dataloader(
        train_path,                                    # 训练数据路径
        imgsz,                                        # 图像尺寸
        batch_size // WORLD_SIZE,                     # 每个进程的批次大小
        gs,                                           # 网格大小，用于数据对齐
        single_cls,                                   # 是否为单类检测
        hyp=hyp,                                      # 超参数字典
        augment=True,                                 # 启用数据增强
        cache=None if opt.cache == 'val' else opt.cache,  # 数据缓存策略
        rect=opt.rect,                                # 是否使用矩形训练
        rank=LOCAL_RANK,                              # 当前进程的本地排名
        workers=workers,                              # 数据加载工作进程数
        image_weights=opt.image_weights,              # 是否使用图像权重采样
        quad=opt.quad,                                # 是否使用quad数据加载
        prefix=colorstr('train: '),                   # 日志前缀
        shuffle=True,                                 # 是否打乱数据
        seed=opt.seed                                 # 随机种子
    )
    
    # ================================== 第十八步：数据集标签验证 ==================================
    labels = np.concatenate(dataset.labels, 0)  # 合并所有标签
    mlc = int(labels[:, 0].max())  # 获取最大标签类别索引
    # 验证标签类别是否超出定义的类别数量
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # ================================== 第十九步：验证数据加载器创建（仅主进程） ==================================
    if RANK in {-1, 0}:
        # 创建验证数据加载器，批次大小为训练的2倍，提高验证效率
        val_loader = create_dataloader(
            val_path,                                 # 验证数据路径
            imgsz,                                    # 图像尺寸
            batch_size // WORLD_SIZE * 2,            # 验证批次大小（是训练的2倍）
            gs,                                       # 网格大小
            single_cls,                               # 是否为单类检测
            hyp=hyp,                                  # 超参数字典
            cache=None if noval else opt.cache,       # 缓存策略：不验证时不缓存
            rect=True,                                # 验证时使用矩形推理
            rank=-1,                                  # 验证时不使用分布式
            workers=workers * 2,                      # 验证时使用更多工作进程
            pad=0.5,                                  # 填充比例
            prefix=colorstr('val: ')                  # 日志前缀
        )[0]

        # ================================== 第二十步：模型初始化检查 ==================================
        if not resume:  # 如果不是恢复训练，进行初始化检查
            if not opt.noautoanchor:
                # 自动锚框优化：根据数据集特征优化锚框尺寸
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            # 预处理：先转半精度再转回全精度，用于锚框精度优化
            model.half().float()  # pre-reduce anchor precision

        # 触发预训练结束回调，传递标签和类别名称
        callbacks.run('on_pretrain_routine_end', labels, names)

    # ================================== 第二十一步：分布式数据并行（DDP）模式配置 ==================================
    # 在CUDA环境下且不是单GPU模式时，启用智能DDP
    if cuda and RANK != -1:
        model = smart_DDP(model)  # 智能分布式数据并行配置

    # ================================== 第二十二步：模型属性配置和损失函数缩放 ==================================
    # 获取检测层数量，用于损失函数缩放
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    
    # 根据检测层数量调整损失函数权重，确保不同架构间的一致性
    hyp['box'] *= 3 / nl                          # 边界框损失权重：缩放到检测层数
    hyp['cls'] *= nc / 80 * 3 / nl                # 分类损失权重：根据类别数和检测层数缩放
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl     # 置信度损失权重：根据图像尺寸和检测层数缩放
    hyp['label_smoothing'] = opt.label_smoothing  # 标签平滑参数
    
    # 将关键属性附加到模型对象上，便于后续访问
    model.nc = nc                                 # 类别数量
    model.hyp = hyp                              # 超参数字典
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 类别权重
    model.names = names                          # 类别名称

    # ================================== 第二十三步：训练初始化设置 ==================================
    t0 = time.time()                             # 记录训练开始时间
    nb = len(train_loader)                       # 训练批次总数
    # 计算预热迭代次数：至少100次，最多为warmup_epochs * batch_count
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training - 可选：限制预热时长
    
    # 训练状态变量初始化
    last_opt_step = -1                           # 上次优化器更新的步数
    maps = np.zeros(nc)                          # 每个类别的mAP初始化
    results = (0, 0, 0, 0, 0, 0, 0)              # 评估结果：P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1       # 设置学习率调度器的起始epoch
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 混合精度梯度缩放器
    stopper, stop = EarlyStopping(patience=opt.patience), False  # 早停机制初始化
    compute_loss = ComputeLoss(model)            # 初始化损失计算类
    
    # 触发训练开始回调
    callbacks.run('on_train_start')
    
    # 打印训练信息摘要
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    # ================================== 第二十四步：主训练循环 ==================================
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # 触发epoch开始回调
        callbacks.run('on_train_epoch_start')
        model.train()  # 设置模型为训练模式

        # ================================== Epoch级别的数据处理 ==================================
        # 图像权重更新（可选，仅单GPU模式）
        if opt.image_weights:
            # 根据类别权重和mAP计算图像采样权重，提高困难样本的采样概率
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Mosaic边界更新（可选，用于数据增强）
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # ================================== Batch循环初始化 ==================================
        mloss = torch.zeros(3, device=device)  # 平均损失初始化：[box_loss, obj_loss, cls_loss]
        # 分布式训练中设置每个epoch的随机种子
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        # 创建进度条和日志显示
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar - 仅主进程显示进度条
        
        optimizer.zero_grad()  # 清零梯度

        # ================================== 批次训练循环 ==================================
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')  # 触发批次开始回调
            ni = i + nb * epoch  # 自训练开始的累计批次数
            # 图像预处理：转移到设备，转换数据类型，归一化到[0,1]
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # ================================== 预热阶段处理 ==================================
            if ni <= nw:  # 在预热阶段内
                xi = [0, nw]  # 插值区间
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                # 动态调整梯度累积步数
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                
                # 动态调整学习率和动量
                for j, x in enumerate(optimizer.param_groups):
                    # bias参数的学习率从0.1降到lr0，其他参数从0.0升到lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        # 动量从warmup_momentum渐变到目标momentum
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # ================================== 多尺度训练 ==================================
            if opt.multi_scale:
                # 随机选择训练尺寸，增强模型对不同尺寸的适应性
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor - 缩放因子
                if sf != 1:
                    # 双线性插值调整图像尺寸
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # ================================== 前向传播 ==================================
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=amp):  # 自动混合精度前向传播
                pred = model(imgs)  # forward - 模型前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 计算损失，已按batch_size缩放
                if RANK != -1:
                    loss *= WORLD_SIZE  # 分布式训练中梯度在设备间平均
                if opt.quad:
                    loss *= 4.  # quad数据加载时的损失调整

            # ================================== 反向传播 ==================================
            scaler.scale(loss).backward()  # 缩放损失后反向传播

            # ================================== 优化器更新 ==================================
            # 梯度累积：当累积足够的梯度时才更新参数
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # 取消梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪，防止梯度爆炸
                scaler.step(optimizer)  # 执行优化器步骤
                scaler.update()  # 更新缩放因子
                optimizer.zero_grad()  # 清零梯度
                if ema:
                    ema.update(model)  # 更新EMA模型
                last_opt_step = ni  # 记录最后一次优化步骤

            # ================================== 日志记录和进度显示 ==================================
            if RANK in {-1, 0}:  # 仅主进程记录日志
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新移动平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # GPU内存使用量(GB)
                # 更新进度条显示：epoch/总epoch, 内存使用, 三种损失, 目标数量, 图像尺寸
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:  # 检查是否需要停止训练
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # ================================== Epoch结束后的处理 ==================================
        # 学习率调度器更新
        lr = [x['lr'] for x in optimizer.param_groups]  # 获取当前学习率，用于日志记录
        scheduler.step()  # 执行学习率调度步骤

        # ================================== 验证和评估（仅主进程） ==================================
        if RANK in {-1, 0}:
            # 触发epoch结束回调
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 更新EMA模型的属性，确保与主模型同步
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断是否为最后一个epoch或可能触发早停
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            
            # 执行验证：如果不禁用验证或到达最后epoch
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,                              # 数据集配置
                    batch_size=batch_size // WORLD_SIZE * 2, # 验证批次大小（训练的2倍）
                    imgsz=imgsz,                           # 图像尺寸
                    half=amp,                              # 是否使用半精度
                    model=ema.ema,                         # 使用EMA模型进行验证
                    single_cls=single_cls,                 # 是否单类检测
                    dataloader=val_loader,                 # 验证数据加载器
                    save_dir=save_dir,                     # 结果保存目录
                    plots=False,                           # 不生成验证图表
                    callbacks=callbacks,                   # 回调函数
                    compute_loss=compute_loss              # 损失计算函数
                )

            # ================================== 最佳模型更新和早停检查 ==================================
            # 计算综合适应度分数：精度、召回率、mAP的加权组合
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)          # 检查是否需要早停
            # 更新最佳适应度
            if fi > best_fitness:
                best_fitness = fi
            # 准备日志记录数据
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # ================================== 模型保存 ==================================
            # 保存模型：不禁用保存 或 (最后epoch且不是进化模式)
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 构建检查点字典，包含完整的训练状态
                ckpt = {
                    'epoch': epoch,                                    # 当前epoch
                    'best_fitness': best_fitness,                     # 最佳适应度
                    'model': deepcopy(de_parallel(model)).half(),     # 模型权重（半精度）
                    'ema': deepcopy(ema.ema).half(),                  # EMA模型权重（半精度）
                    'updates': ema.updates,                           # EMA更新次数
                    'optimizer': optimizer.state_dict(),              # 优化器状态
                    'opt': vars(opt),                                 # 训练选项
                    'git': GIT_INFO,                                  # Git信息：{remote, branch, commit}
                    'date': datetime.now().isoformat()               # 保存时间戳
                }

                # 执行模型保存策略
                torch.save(ckpt, last)                               # 保存最新模型
                if best_fitness == fi:                               # 如果是最佳模型
                    torch.save(ckpt, best)                           # 保存最佳模型
                if opt.save_period > 0 and epoch % opt.save_period == 0:  # 定期保存
                    torch.save(ckpt, w / f'epoch{epoch}.pt')         # 保存特定epoch的模型
                del ckpt  # 清理内存
                # 触发模型保存回调
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # ================================== 早停机制（分布式同步） ==================================
        if RANK != -1:  # 如果是分布式训练
            # 将主进程的早停决策广播到所有进程
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # 广播早停信号到所有rank
            if RANK != 0:
                stop = broadcast_list[0]  # 从广播中获取早停信号
        if stop:
            break  # 所有DDP进程必须同时退出

        # end epoch ----------------------------------------------------------------------------------------------------
    
    # ================================== 训练结束后的处理 ==================================
    if RANK in {-1, 0}:  # 仅主进程执行
        # 输出训练完成信息
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        
        # 对保存的模型进行最终处理
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 清除优化器信息，减小文件大小
                if f is best:  # 对最佳模型进行最终验证
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,                                # 数据集配置
                        batch_size=batch_size // WORLD_SIZE * 2,  # 验证批次大小
                        imgsz=imgsz,                             # 图像尺寸
                        model=attempt_load(f, device).half(),    # 加载最佳模型
                        iou_thres=0.65 if is_coco else 0.60,    # IoU阈值：COCO用0.65，其他用0.60
                        single_cls=single_cls,                   # 是否单类检测
                        dataloader=val_loader,                   # 验证数据加载器
                        save_dir=save_dir,                       # 保存目录
                        save_json=is_coco,                       # COCO数据集保存JSON结果
                        verbose=True,                            # 详细输出
                        plots=plots,                             # 生成验证图表
                        callbacks=callbacks,                     # 回调函数
                        compute_loss=compute_loss                # 损失计算函数
                    )  # 验证最佳模型并生成图表
                    if is_coco:  # 如果是COCO数据集，触发额外的回调
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        # 触发训练结束回调
        callbacks.run('on_train_end', last, best, epoch, results)

    # 清理GPU缓存
    torch.cuda.empty_cache()
    return results  # 返回最终的验证结果


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    YOLOv5训练主入口函数
    
    功能描述：
    这是YOLOv5训练的核心控制函数，负责整个训练流程的协调和管理，
    包括参数验证、恢复训练、分布式设置、普通训练和超参数进化等功能。
    
    @param opt: 命令行参数对象，包含所有训练配置
    @param callbacks: 回调函数管理器，用于训练过程中的事件处理
    """
    # ================================== 第一步：基础检查（仅主进程执行） ==================================
    if RANK in {-1, 0}:  # 只有主进程（单机训练或分布式训练的rank=0进程）执行检查
        print_args(vars(opt))    # 打印所有训练参数，便于调试和记录
        check_git_status()       # 检查Git仓库状态，确保代码版本一致性
        check_requirements()     # 检查Python依赖包是否满足要求

    # ================================== 第二步：恢复训练处理 ==================================
    # 当用户指定--resume且不是Comet实验恢复且不是超参数进化模式时，进入恢复训练流程
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 确定检查点文件路径：用户指定的路径 或 最近的训练结果
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        
        # 构建配置文件路径：检查点文件的上级目录中的opt.yaml
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # 备份原始数据集路径，避免被覆盖
        
        # 尝试从YAML文件加载训练配置
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)  # 从YAML文件加载配置字典
        else:
            # 如果YAML文件不存在，从检查点文件中提取配置
            d = torch.load(last, map_location='cpu', weights_only=False)['opt']
        
        # 重建训练参数对象
        opt = argparse.Namespace(**d)  # 用历史配置替换当前配置
        # 重新设置关键路径参数
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        
        # 处理在线数据集URL，避免HUB认证超时问题
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
            
    # ================================== 第三步：新训练的参数验证和路径设置 ==================================
    else:
        # 验证和规范化所有输入文件路径
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        
        # 确保用户至少指定了配置文件或预训练权重之一
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        
        # ================================== 超参数进化模式的特殊设置 ==================================
        if opt.evolve:
            # 如果使用默认项目名，重命名为进化专用目录
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            # 进化模式下的特殊设置：允许目录存在，禁用恢复训练
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        
        # 如果实验名称为'cfg'，使用模型配置文件名作为实验名
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
            
        # 生成唯一的保存目录路径，避免覆盖已有实验
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # ================================== 第四步：设备选择和分布式训练设置 ==================================
    # 智能选择训练设备（CPU或GPU）
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # 分布式数据并行（DDP）模式配置
    if LOCAL_RANK != -1:  # 如果LOCAL_RANK != -1，说明处于分布式训练模式
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        
        # ================================== 分布式训练兼容性检查 ==================================
        # 检查各种功能与DDP的兼容性，不兼容的功能会抛出异常
        assert not opt.image_weights, f'--image-weights {msg}'  # 图像权重采样与DDP不兼容
        assert not opt.evolve, f'--evolve {msg}'                # 超参数进化与DDP不兼容
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'  # 自动批次大小与DDP不兼容
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'  # 批次大小必须能被总进程数整除
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'  # 检查CUDA设备数量是否足够
        
        # ================================== DDP进程组初始化 ==================================
        torch.cuda.set_device(LOCAL_RANK)               # 设置当前进程使用的GPU设备
        device = torch.device('cuda', LOCAL_RANK)        # 创建对应的设备对象
        # 初始化进程组：优先使用NCCL后端（GPU通信更高效），备选GLOO后端
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # ================================== 第五步：执行训练 ==================================
    # 普通训练模式：直接调用训练函数
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)  # 执行单次训练

    # ================================== 第六步：超参数进化算法（可选） ==================================
    else:
        # 超参数进化元数据：定义每个超参数的变异范围和约束
        # 格式：(变异增益, 下限, 上限)
        meta = {
            # ================================== 优化器相关参数 ==================================
            'lr0': (1, 1e-5, 1e-1),              # 初始学习率 (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),               # 最终学习率因子 (OneCycleLR: lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),        # SGD动量/Adam beta1参数
            'weight_decay': (1, 0.0, 0.001),     # 优化器权重衰减
            
            # ================================== 学习率预热相关参数 ==================================
            'warmup_epochs': (1, 0.0, 5.0),     # 预热epoch数（支持小数）
            'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
            'warmup_bias_lr': (1, 0.0, 0.2),    # 预热偏置学习率
            
            # ================================== 损失函数权重参数 ==================================
            'box': (1, 0.02, 0.2),              # 边界框损失权重
            'cls': (1, 0.2, 4.0),               # 分类损失权重
            'cls_pw': (1, 0.5, 2.0),            # 分类BCE损失正样本权重
            'obj': (1, 0.2, 4.0),               # 置信度损失权重（随像素缩放）
            'obj_pw': (1, 0.5, 2.0),            # 置信度BCE损失正样本权重
            
            # ================================== 训练策略参数 ==================================
            'iou_t': (0, 0.1, 0.7),             # IoU训练阈值
            'anchor_t': (1, 2.0, 8.0),          # 锚框倍数阈值
            'anchors': (2, 2.0, 10.0),          # 每个输出网格的锚框数（0表示忽略）
            'fl_gamma': (0, 0.0, 2.0),          # focal loss gamma参数（EfficientDet默认1.5）
            
            # ================================== 数据增强参数 ==================================
            # HSV颜色空间增强
            'hsv_h': (1, 0.0, 0.1),             # 色调增强强度
            'hsv_s': (1, 0.0, 0.9),             # 饱和度增强强度
            'hsv_v': (1, 0.0, 0.9),             # 亮度增强强度
            
            # 几何变换增强
            'degrees': (1, 0.0, 45.0),          # 图像旋转角度（±度）
            'translate': (1, 0.0, 0.9),         # 图像平移范围（±比例）
            'scale': (1, 0.0, 0.9),             # 图像缩放范围（±增益）
            'shear': (1, 0.0, 10.0),            # 图像剪切角度（±度）
            'perspective': (0, 0.0, 0.001),     # 透视变换强度（±比例，范围0-0.001）
            
            # 翻转和混合增强
            'flipud': (1, 0.0, 1.0),            # 上下翻转概率
            'fliplr': (0, 0.0, 1.0),            # 左右翻转概率
            'mosaic': (1, 0.0, 1.0),            # Mosaic数据增强概率
            'mixup': (1, 0.0, 1.0),             # Mixup数据增强概率
            'copy_paste': (1, 0.0, 1.0)         # 分割复制粘贴增强概率
        }

        # ================================== 超参数初始化 ==================================
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载基础超参数字典
            if 'anchors' not in hyp:  # 如果超参数中没有锚框设置
                hyp['anchors'] = 3   # 设置默认锚框数量
        
        # 如果禁用自动锚框，从进化参数中移除锚框相关设置
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
            
        # 进化模式的特殊设置：只在最后epoch验证和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        
        # 定义进化结果文件路径
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        
        # 如果指定了云存储桶，尝试下载已有的进化结果
        if opt.bucket:
            # download evolve.csv if exists - 下载已存在的进化CSV文件
            subprocess.run([
                'gsutil',                          # Google Cloud Storage工具
                'cp',                              # 复制命令
                f'gs://{opt.bucket}/evolve.csv',   # 云端文件路径
                str(evolve_csv),                   # 本地目标路径
            ])

        # ================================== 进化主循环 ==================================
        for _ in range(opt.evolve):  # 执行指定数量的进化代数
            if evolve_csv.exists():  # 如果进化历史文件存在：选择最优超参数并变异
                # ================================== 父代选择 ==================================
                parent = 'single'  # 父代选择方法：'single'(单一) 或 'weighted'(加权)
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)  # 加载历史进化结果
                n = min(5, len(x))  # 考虑的历史结果数量（最多5个）
                x = x[np.argsort(-fitness(x))][:n]  # 按适应度降序排序，取前n个
                w = fitness(x) - fitness(x).min() + 1E-6  # 计算选择权重（确保和>0）
                
                if parent == 'single' or len(x) == 1:
                    # 单一父代选择：基于适应度权重随机选择一个父代
                    # x = x[random.randint(0, n - 1)]  # random selection - 随机选择（已注释）
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection - 加权选择
                elif parent == 'weighted':
                    # 加权父代选择：多个父代的加权组合
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # ================================== 变异操作 ==================================
                mp, s = 0.8, 0.2  # 变异概率=0.8, 标准差=0.2
                npr = np.random   # NumPy随机数生成器
                npr.seed(int(time.time()))  # 使用当前时间作为随机种子
                g = np.array([meta[k][0] for k in hyp.keys()])  # 提取每个参数的增益因子
                ng = len(meta)    # 参数总数
                v = np.ones(ng)   # 变异向量初始化
                
                # 生成变异向量，确保至少有一个参数发生变化
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    # 生成变异向量：增益 × 是否变异 × 正态分布噪声 × 随机缩放 + 1
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                
                # 应用变异到各个超参数
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])  # mutate - 原值乘以变异因子

            # ================================== 参数约束 ==================================
            # 将所有超参数限制在预定义的合法范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])    # 应用下限约束
                hyp[k] = min(hyp[k], v[2])    # 应用上限约束
                hyp[k] = round(hyp[k], 5)     # 保留5位有效数字

            # ================================== 训练变异体 ==================================
            results = train(hyp.copy(), opt, device, callbacks)  # 使用变异后的超参数训练
            callbacks = Callbacks()  # 重置回调函数管理器
            
            # ================================== 记录变异结果 ==================================
            # 定义要记录的关键指标
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            # 打印并保存变异结果
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # ================================== 进化结果分析 ==================================
        plot_evolve(evolve_csv)  # 生成进化过程可视化图表
        
        # 输出进化完成信息和使用示例
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
