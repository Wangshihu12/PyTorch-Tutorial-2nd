# -*- coding:utf-8 -*-
"""
@file name  : 07_lenet_ensemble.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-07-06
@brief      : 编写自定义的模型集成类 - 实现基于投票机制的深度学习模型集成
             
             模型集成是一种通过组合多个基础模型的预测结果来提高整体性能的技术
             本代码实现了基于torchensemble框架的自定义集成类，支持多个相同架构但不同初始化的模型
             通过投票机制进行最终预测，通常能获得比单个模型更好的性能
"""
import os
import torch
import torchvision
import torchmetrics
import torch.nn as nn
import my_utils as utils
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchensemble.utils import set_module
from torchensemble.voting import VotingClassifier


# =============================== 类别定义 ===============================
# CIFAR-10数据集的10个类别名称
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_args_parser(add_help=True):
    """
    创建命令行参数解析器
    
    定义了训练脚本所需的所有超参数和配置选项
    
    @param add_help: bool, 是否添加帮助信息，默认为True
    @return: argparse.ArgumentParser, 配置好的参数解析器
    """
    import argparse

    # 创建参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    # =============================== 数据相关参数 ===============================
    # 数据集路径
    parser.add_argument("--data-path", default=r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office", type=str, help="dataset path")
    
    # 模型名称
    parser.add_argument("--model", default="resnet8", type=str, help="model name")
    
    # 计算设备选择
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    # 批次大小（每个GPU的样本数）
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    
    # 总训练轮数
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    
    # 数据加载的工作进程数
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )

    # =============================== 优化器相关参数 ===============================
    # 优化器类型
    parser.add_argument("--opt", default="SGD", type=str, help="optimizer")
    
    # 随机种子，确保实验可重复性
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    
    # 初始学习率
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    
    # 动量参数（SGD优化器使用）
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    
    # 权重衰减（L2正则化）
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # =============================== 学习率调度参数 ===============================
    # 学习率衰减步长（每隔多少个epoch衰减一次）
    parser.add_argument("--lr-step-size", default=80, type=int, help="decrease lr every step-size epochs")
    
    # 学习率衰减因子（每次衰减乘以的系数）
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    # =============================== 输出和日志参数 ===============================
    # 打印频率（每隔多少个iteration打印一次）
    parser.add_argument("--print-freq", default=80, type=int, help="print frequency")
    
    # 输出目录路径
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    
    # 恢复训练的检查点路径
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    
    # 开始训练的epoch（用于恢复训练）
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    return parser


def main(args):
    """
    主训练函数
    
    执行完整的训练流程，包括数据准备、模型构建、集成训练和保存
    
    @param args: 解析后的命令行参数
    """
    # 提取关键参数
    device = args.device          # 计算设备
    data_dir = args.data_path     # 数据目录
    result_dir = args.output_dir  # 结果保存目录
    
    # =============================== 日志和可视化设置 ===============================
    # 创建日志记录器和日志目录
    logger, log_dir = utils.make_logger(result_dir)
    
    # 创建TensorBoard写入器，用于记录训练过程
    writer = SummaryWriter(log_dir=log_dir)

    # =============================== 步骤1：数据集准备 ===============================
    
    # CIFAR-10数据集的标准化参数（基于训练集计算得出）
    normMean = [0.4948052, 0.48568845, 0.44682974]    # RGB三通道的均值
    normStd = [0.24580306, 0.24236229, 0.2603115]     # RGB三通道的标准差
    
    # 创建标准化变换
    normTransform = transforms.Normalize(normMean, normStd)
    
    # 训练集的数据增强变换
    train_transform = transforms.Compose([
        transforms.Resize(32),                    # 调整图像尺寸为32x32
        transforms.RandomCrop(32, padding=4),    # 随机裁剪，增加数据多样性
        transforms.ToTensor(),                    # 转换为PyTorch张量
        normTransform                             # 应用标准化
    ])

    # 验证集的变换（不需要数据增强）
    valid_transform = transforms.Compose([
        transforms.ToTensor(),                    # 转换为PyTorch张量
        normTransform                             # 应用标准化
    ])

    # 加载CIFAR-10数据集
    # 注意：root变量下需要存放cifar-10-python.tar.gz文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, num_workers=args.workers)

    # =============================== 步骤2：模型构建和集成 ===============================
    # 创建基础模型（ResNet20）
    model_base = utils.resnet20()
    # model_base = utils.LeNet5()  # 也可以使用LeNet5作为基础模型
    
    # 创建集成模型实例
    # estimator: 基础模型类型
    # n_estimators: 集成中模型的数量（这里设置为3）
    # 其他参数：日志记录器、设备、参数配置、类别、TensorBoard写入器、保存目录
    model = MyEnsemble(estimator=model_base, n_estimators=3, logger=logger, device=device, args=args,
                       classes=classes, writer=writer, save_dir=log_dir)
    
    # 设置优化器参数
    model.set_optimizer(args.opt, lr=args.lr, weight_decay=args.weight_decay)
    
    # 开始训练
    model.fit(train_loader, test_loader=valid_loader, epochs=args.epochs)


class MyEnsemble(VotingClassifier):
    """
    自定义模型集成类
    
    继承自torchensemble.voting.VotingClassifier，实现了基于投票机制的模型集成
    支持多个相同架构但不同初始化的模型，通过投票进行最终预测
    
    特点：
    1. 支持多个基础模型的同时训练
    2. 每个模型有独立的优化器和学习率调度器
    3. 支持模型保存和恢复
    4. 完整的训练监控和日志记录
    """
    
    def __init__(self, **kwargs):
        """
        初始化集成模型
        
        @param kwargs: 包含以下键值对的字典：
            - estimator: 基础模型类型
            - n_estimators: 集成中模型的数量
            - logger: 日志记录器
            - writer: TensorBoard写入器
            - device: 计算设备
            - args: 训练参数
            - classes: 类别列表
            - save_dir: 模型保存目录
        """
        # 调用父类VotingClassifier的初始化方法
        # 传入基础模型类型和模型数量
        super(VotingClassifier, self).__init__(kwargs["estimator"], kwargs["n_estimators"])
        
        # 保存传入的参数
        self.logger = kwargs["logger"]      # 日志记录器
        self.writer = kwargs["writer"]      # TensorBoard写入器
        self.device = kwargs["device"]      # 计算设备
        self.args = kwargs["args"]          # 训练参数
        self.classes = kwargs["classes"]    # 类别列表
        self.save_dir = kwargs["save_dir"]  # 模型保存目录

    @staticmethod
    def save(model, save_dir, logger):
        """
        实现模型序列化到指定目录
        
        保存集成模型的状态，包括所有基础模型、优化器状态等
        
        @param model: MyEnsemble, 要保存的集成模型
        @param save_dir: str, 保存目录路径
        @param logger: Logger, 日志记录器
        """
        # 如果没有指定保存目录，使用当前目录
        if save_dir is None:
            save_dir = "./"

        # 如果保存目录不存在，创建它
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # 确定基础估计器的名称
        # 检查base_estimator_是类型还是实例
        if isinstance(model.base_estimator_, type):
            base_estimator_name = model.base_estimator_.__name__
        else:
            base_estimator_name = model.base_estimator_.__class__.__name__

        # 构建文件名：{集成模型名称}_{基础估计器名称}_{估计器数量}_ckpt.pth
        # 例如：MyEnsemble_ResNet20_3_ckpt.pth
        filename = "{}_{}_{}_ckpt.pth".format(
            type(model).__name__,           # 集成模型类名
            base_estimator_name,            # 基础估计器名称
            model.n_estimators,             # 估计器数量
        )

        # 构建保存状态字典
        # 注意：某些集成方法中，实际的估计器数量可能与n_estimators不同
        state = {
            "n_estimators": len(model.estimators_),  # 实际的估计器数量
            "model": model.state_dict(),             # 模型状态字典
            "_criterion": model._criterion,          # 损失函数
        }
        
        # 构建完整的保存路径
        save_dir = os.path.join(save_dir, filename)

        # 记录保存信息
        logger.info("Saving the model to `{}`".format(save_dir))

        # 保存模型
        torch.save(state, save_dir)

        return

    def fit(self, train_loader, epochs=100, log_interval=100, test_loader=None, save_model=True, save_dir=None):
        """
        训练集成模型
        
        训练所有基础模型，每个模型有独立的优化器和学习率调度器
        
        @param train_loader: DataLoader, 训练数据加载器
        @param epochs: int, 训练轮数，默认100
        @param log_interval: int, 日志记录间隔，默认100
        @param test_loader: DataLoader, 测试数据加载器，用于验证
        @param save_model: bool, 是否保存模型，默认True
        @param save_dir: str, 模型保存目录，默认None
        """
        # =============================== 创建模型、优化器、学习率调度器和评估器列表 ===============================
        
        # 创建基础估计器列表
        estimators = []
        for _ in range(self.n_estimators):
            # 为每个估计器创建独立的模型实例
            estimators.append(self._make_estimator())

        # 创建优化器列表
        optimizers = []
        schedulers = []
        for i in range(self.n_estimators):
            # 为每个估计器创建独立的优化器
            optimizers.append(set_module.set_optimizer(estimators[i],
                                                       self.optimizer_name, **self.optimizer_args))
            
            # 为每个估计器创建独立的学习率调度器
            # 使用MultiStepLR：在第100和150个epoch时降低学习率
            scheduler_ = torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[100, 150],
                                                              gamma=self.args.lr_gamma)
            
            # 注释掉的StepLR调度器（每固定步数降低学习率）
            # scheduler_ = torch.optim.lr_scheduler.StepLR(optimizers[i], step_size=self.args.lr_step_size,
            #                                             gamma=self.args.lr_gamma)
            
            schedulers.append(scheduler_)

        # 创建准确率评估器列表
        acc_metrics = []
        for i in range(self.n_estimators):
            acc_metrics.append(torchmetrics.Accuracy())

        # 设置损失函数
        self._criterion = nn.CrossEntropyLoss()

        # =============================== 开始训练循环 ===============================
        # 记录最佳验证准确率
        best_acc = 0.
        
        # 遍历所有训练轮数
        for epoch in range(epochs):
            # =============================== 训练阶段 ===============================
            # 依次训练每个基础模型
            for model_idx, (estimator, optimizer, scheduler) in enumerate(zip(estimators, optimizers, schedulers)):
                # 训练一个epoch，返回损失、准确率和混淆矩阵
                loss_m_train, acc_m_train, mat_train = \
                    utils.ModelTrainerEnsemble.train_one_epoch(
                        train_loader, estimator, self._criterion, optimizer, scheduler, epoch,
                        self.device, self.args, self.logger, self.classes)
                
                # 更新学习率
                scheduler.step()

                # =============================== TensorBoard记录 ===============================
                # 记录每个模型的训练损失
                self.writer.add_scalars('Loss_group', {'train_loss_{}'.format(model_idx):
                                                           loss_m_train.avg}, epoch)
                
                # 记录每个模型的训练准确率
                self.writer.add_scalars('Accuracy_group', {'train_acc_{}'.format(model_idx):
                                                               acc_m_train.avg}, epoch)
                
                # 记录学习率
                self.writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

            # =============================== 验证阶段 ===============================
            # 在验证集上评估所有模型的集成性能
            loss_valid_meter, acc_valid, top1_group, mat_valid = \
                utils.ModelTrainerEnsemble.evaluate(test_loader, estimators, self._criterion, self.device, self.classes)

            # =============================== 日志记录 ===============================
            # 记录验证损失
            self.writer.add_scalars('Loss_group', {'valid_loss':
                                                       loss_valid_meter.avg}, epoch)
            
            # 记录验证准确率
            self.writer.add_scalars('Accuracy_group', {'valid_acc':
                                                           acc_valid*100}, epoch)

            # 记录详细的训练信息
            self.logger.info(
                'Epoch: [{:0>3}/{:0>3}]  '
                'Train Loss avg: {loss_train:>6.4f}  '
                'Valid Loss avg: {loss_valid:>6.4f}  '
                'Train Acc@1 avg:  {top1_train:>7.2f}%   '
                'Valid Acc@1 avg: {top1_valid:>7.2%}    '
                'LR: {lr}'.format(
                    epoch, self.args.epochs, loss_train=loss_m_train.avg, loss_valid=loss_valid_meter.avg,
                    top1_train=acc_m_train.avg, top1_valid=acc_valid, lr=schedulers[0].get_last_lr()[0]))

            # 记录每个模型的验证准确率
            for model_idx, top1_meter in enumerate(top1_group):
                self.writer.add_scalars('Accuracy_group', {'valid_acc_{}'.format(model_idx): top1_meter.compute()*100}, epoch)

            # =============================== 模型保存 ===============================
            # 如果当前验证准确率超过历史最佳，保存模型
            if acc_valid > best_acc:
                best_acc = acc_valid
                
                # 创建新的ModuleList来存储最佳模型
                self.estimators_ = nn.ModuleList()
                self.estimators_.extend(estimators)
                
                # 如果启用保存，则保存模型
                if save_model:
                    self.save(self, self.save_dir, self.logger)


if __name__ == "__main__":
    # =============================== 主程序入口 ===============================
    # 解析命令行参数
    args = get_args_parser().parse_args()
    
    # 设置随机种子，确保实验可重复性
    utils.setup_seed(args.random_seed)
    
    # 自动检测并设置计算设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 开始主训练流程
    main(args)




