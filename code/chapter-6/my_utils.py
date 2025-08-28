# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    """
    权重初始化函数
    
    使用Kaiming初始化方法对线性层和卷积层的权重进行初始化
    这种方法特别适合使用ReLU激活函数的深度神经网络
    
    @param m: torch.nn.Module, 需要初始化的模块
    """
    classname = m.__class__.__name__
    # 只对线性层和卷积层进行权重初始化
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # 使用Kaiming正态分布初始化，适合ReLU激活函数
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """
    Lambda层：包装任意函数作为神经网络层
    
    这个类允许我们将任意Python函数包装成PyTorch模块
    主要用于实现复杂的shortcut连接
    """
    def __init__(self, lambd):
        """
        初始化Lambda层
        
        @param lambd: function, 要包装的函数
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        """
        前向传播：直接调用包装的函数
        
        @param x: torch.Tensor, 输入张量
        @return: torch.Tensor, 函数处理后的结果
        """
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    ResNet的基本残差块
    
    这是ResNet的核心组件，包含两个3x3卷积层和一个shortcut连接
    支持不同的shortcut选项来处理维度不匹配的情况
    """
    expansion = 1  # 扩展因子，BasicBlock中为1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        """
        初始化基本残差块
        
        @param in_planes: int, 输入通道数
        @param planes: int, 输出通道数
        @param stride: int, 第一个卷积层的步长
        @param option: str, shortcut连接的处理方式，'A'或'B'
        """
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层：3x3卷积，可能改变空间尺寸和通道数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        
        # 第二个卷积层：3x3卷积，保持空间尺寸和通道数
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化

        # 初始化shortcut连接
        self.shortcut = nn.Sequential()
        
        # 当需要改变维度或步长时，需要特殊的shortcut处理
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                CIFAR10 ResNet论文使用的选项A
                使用Lambda层实现复杂的shortcut连接
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                # 解释：
                # x[:, :, ::2, ::2]: 在空间维度上采样（步长为2）
                # F.pad: 在通道维度上填充，使通道数匹配
                # planes//4: 计算需要填充的通道数
            elif option == 'B':
                # 选项B：使用1x1卷积和批归一化
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        """
        前向传播：实现残差连接
        
        @param x: torch.Tensor, 输入张量
        @return: torch.Tensor, 残差块的输出
        """
        # 主路径：两个卷积层
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积+批归一化+ReLU
        out = self.bn2(self.conv2(out))         # 第二个卷积+批归一化
        
        # 残差连接：主路径输出 + shortcut输入
        out += self.shortcut(x)
        
        # 最终激活
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet网络主体
    
    实现了完整的ResNet架构，支持不同深度的配置
    专门为CIFAR-10等小尺寸图像设计
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        初始化ResNet
        
        @param block: class, 残差块类型（如BasicBlock）
        @param num_blocks: list, 每个阶段的残差块数量
        @param num_classes: int, 分类类别数
        """
        super(ResNet, self).__init__()
        self.in_planes = 16  # 初始通道数

        # 第一个卷积层：3x3卷积，将3通道输入转换为16通道
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 构建三个残差阶段
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)   # 16通道，不改变空间尺寸
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)   # 32通道，空间尺寸减半
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)   # 64通道，空间尺寸减半
        
        # 分类头：全局平均池化 + 全连接层
        self.linear = nn.Linear(64, num_classes)

        # 应用权重初始化
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        构建一个残差阶段
        
        @param block: class, 残差块类型
        @param planes: int, 该阶段的通道数
        @param num_blocks: int, 该阶段的残差块数量
        @param stride: int, 第一个残差块的步长
        @return: nn.Sequential, 包含多个残差块的序列
        """
        # 第一个块使用指定的步长，后续块使用步长1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        # 构建该阶段的所有残差块
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # 更新输入通道数，考虑扩展因子
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        @param x: torch.Tensor, 输入图像，形状为(B, 3, H, W)
        @return: torch.Tensor, 分类输出，形状为(B, num_classes)
        """
        # 第一个卷积层
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 三个残差阶段
        out = self.layer1(out)  # 16通道
        out = self.layer2(out)  # 32通道
        out = self.layer3(out)  # 64通道
        
        # 全局平均池化：将特征图转换为向量
        out = F.avg_pool2d(out, out.size()[3])
        
        # 展平：将特征向量转换为1D
        out = out.view(out.size(0), -1)
        
        # 分类：全连接层
        out = self.linear(out)
        return out


def resnet8(num_classes=10):
    """
    创建ResNet-8模型
    
    @param num_classes: int, 分类类别数
    @return: ResNet, ResNet-8模型实例
    """
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def resnet20(num_classes=10):
    """
    创建ResNet-20模型
    
    @param num_classes: int, 分类类别数
    @return: ResNet, ResNet-20模型实例
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def validate(net, data_loader, set_name, classes_name, device):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    
    该函数用于模型验证，计算各种评估指标
    
    @param net: torch.nn.Module, 要验证的神经网络
    @param data_loader: DataLoader, 数据加载器
    @param set_name: str, 数据集名称，如'valid'、'train'、'test'
    @param classes_name: list, 类别名称列表
    @param device: torch.device, 计算设备
    @return: tuple, (混淆矩阵, 准确率字符串)
    """
    # 设置为评估模式
    net.eval()
    cls_num = len(classes_name)
    # 初始化混淆矩阵
    conf_mat = np.zeros([cls_num, cls_num])

    # 遍历数据加载器中的所有批次
    for data in data_loader:
        inputs, labels = data
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = net(inputs)
        # 分离梯度，节省内存
        outputs.detach_()

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()    # 真实标签
            pre_i = predicted[i].cpu().numpy()  # 预测标签
            conf_mat[cate_i, pre_i] += 1.0     # 在对应位置加1

    # 打印每个类别的详细统计信息
    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i],                    # 类别名称
            np.sum(conf_mat[i, :]),            # 该类别的总样本数
            conf_mat[i, i],                    # 正确分类的样本数
            conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),    # 召回率
            conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))   # 精确率

    # 打印整体准确率
    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_conf_mat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False):
    """
    混淆矩阵绘制并保存图片
    
    该函数将混淆矩阵可视化，支持百分比显示和详细统计信息打印
    
    @param confusion_mat: numpy.ndarray, 混淆矩阵
    @param classes: list or tuple, 类别名称列表
    @param set_name: str, 数据集名称（train/valid/test）
    @param out_dir: str, 图片保存目录
    @param epoch: int, 训练轮数，用于文件名
    @param verbose: bool, 是否打印详细的精度信息
    @param perc: bool, 是否使用百分比显示（适用于类别数过多的情况）
    @return: matplotlib.figure.Figure, 生成的混淆矩阵图
    """
    cls_num = len(classes)

    # 归一化混淆矩阵（按行归一化，每行和为1）
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 根据类别数量动态设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        # 在10-100之间线性插值
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    # 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 设置颜色映射和显示混淆矩阵
    cmap = plt.cm.get_cmap('Greys')  # 使用灰度颜色映射
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    
    # 添加颜色条
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')

    # 设置坐标轴标签和刻度
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # x轴标签旋转60度
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 在混淆矩阵上添加数值标签
    if perc:
        # 百分比模式：计算每列的样本总数，然后计算百分比
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), 
                       va='center', ha='center', color='red', fontsize=10)
    else:
        # 绝对数值模式：直接显示样本数量
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), 
                       va='center', ha='center', color='red', fontsize=10)
    
    # 保存图像
    fig.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()  # 关闭图像以释放内存

    # 如果启用详细模式，打印每个类别的统计信息
    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i],                    # 类别名称
                np.sum(confusion_mat[i, :]),  # 该类别的总样本数
                confusion_mat[i, i],          # 正确分类的样本数
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),    # 召回率（避免除零）
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))   # 精确率（避免除零）

    return fig