
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    """
    DDPM模型训练函数：执行扩散模型的训练过程
    
    功能描述：
        该函数实现DDPM（去噪扩散概率模型）的完整训练流程，包括数据加载、
        模型初始化、优化器设置、学习率调度和训练循环。
    
    参数说明：
        modelConfig: 包含所有训练配置参数的字典
    
    返回值说明：
        无返回值，训练过程中会保存模型检查点到指定目录
    """
    # 设置计算设备
    device = torch.device(modelConfig["device"])  # 获取计算设备（如CUDA GPU）
    
    # 数据集准备：加载CIFAR-10训练数据集
    dataset = CIFAR10(
        root=modelConfig['cifar10_dir'],  # CIFAR-10数据集存储路径
        train=True,                        # 使用训练集
        download=True,                     # 如果数据集不存在则自动下载
        transform=transforms.Compose([     # 数据预处理管道
            transforms.RandomHorizontalFlip(),  # 随机水平翻转：数据增强，提高模型泛化能力
            transforms.ToTensor(),               # 转换为张量：将PIL图像转换为PyTorch张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化：将像素值从[0,1]映射到[-1,1]
        ]))
    
    # 数据加载器：批量加载训练数据
    dataloader = DataLoader(
        dataset, 
        batch_size=modelConfig["batch_size"],  # 批大小：每批处理的图像数量
        shuffle=True,                          # 随机打乱数据：每个epoch重新排列数据顺序
        num_workers=4,                        # 数据加载进程数：并行加载数据提高效率
        drop_last=True,                        # 丢弃最后不完整的批次：保持批次大小一致
        pin_memory=True                       # 固定内存：加速GPU数据传输
    )

    # 模型设置：创建UNet网络和训练器
    net_model = UNet(
        T=modelConfig["T"],                           # 扩散步数：前向扩散过程的总步数
        ch=modelConfig["channel"],                     # 基础通道数：网络的基础特征通道数
        ch_mult=modelConfig["channel_mult"],           # 通道倍数：各层通道数的倍数
        attn=modelConfig["attn"],                      # 注意力层位置：指定哪些层使用自注意力
        num_res_blocks=modelConfig["num_res_blocks"],  # 残差块数量：每个分辨率层的残差块数
        dropout=modelConfig["dropout"]                 # Dropout率：防止过拟合的随机失活率
    ).to(device)  # 将模型移动到指定设备（GPU）
    
    # 加载预训练权重（如果指定）
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    
    # 优化器设置：使用AdamW优化器
    optimizer = torch.optim.AdamW(
        net_model.parameters(),    # 模型参数
        lr=modelConfig["lr"],      # 学习率
        weight_decay=1e-4          # 权重衰减：L2正则化，防止过拟合
    )
    
    # 学习率调度器设置：余弦退火调度器
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,           # 优化器
        T_max=modelConfig["epoch"],     # 最大epoch数：余弦退火的周期
        eta_min=0,                      # 最小学习率：学习率的下界
        last_epoch=-1                   # 起始epoch：-1表示从头开始
    )
    
    # 预热调度器：渐进式学习率预热
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,                           # 优化器
        multiplier=modelConfig["multiplier"],           # 乘数因子：预热期间学习率的倍数
        warm_epoch=modelConfig["epoch"] // 10,          # 预热epoch数：总epoch数的10%
        after_scheduler=cosineScheduler                # 预热后的调度器：余弦退火调度器
    )
    
    # 扩散训练器：封装扩散模型的训练逻辑
    trainer = GaussianDiffusionTrainer(
        net_model,                    # UNet网络
        modelConfig["beta_1"],        # 噪声调度起始值
        modelConfig["beta_T"],        # 噪声调度结束值
        modelConfig["T"]               # 扩散步数
    ).to(device)  # 将训练器移动到指定设备

    # 开始训练循环：遍历所有epoch
    for e in range(modelConfig["epoch"]):
        # 使用tqdm显示训练进度条
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            # 遍历当前epoch的所有数据批次
            for images, labels in tqdmDataLoader:
                # 训练步骤
                optimizer.zero_grad()  # 清零梯度：防止梯度累积
                x_0 = images.to(device)  # 将图像数据移动到GPU
                
                # 计算扩散损失：训练器执行前向传播并计算损失
                loss = trainer(x_0).sum() / 1000.  # 计算损失并除以1000进行缩放
                
                loss.backward()  # 反向传播：计算梯度
                
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                
                optimizer.step()  # 更新模型参数
                
                # 更新进度条显示信息
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,                                                    # 当前epoch
                    "loss: ": loss.item(),                                         # 当前损失值
                    "img shape: ": x_0.shape,                                      # 图像张量形状
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]        # 当前学习率
                })
        
        # 更新学习率调度器
        warmUpScheduler.step()
        
        # 保存模型检查点：每个epoch结束后保存模型权重
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    """
    DDPM模型推理函数：使用训练好的模型生成新图像
    
    功能描述：
        该函数实现DDPM模型的推理过程，从纯噪声开始，通过反向扩散过程
        逐步去噪生成高质量的图像。这是扩散模型的核心应用场景。
    
    参数说明：
        modelConfig: 包含所有推理配置参数的字典
    
    返回值说明：
        无返回值，生成的图像会保存到指定目录
    """
    # 使用torch.no_grad()禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        # 设置计算设备
        device = torch.device(modelConfig["device"])  # 获取计算设备（如CUDA GPU）
        
        # 创建UNet模型：使用与训练时相同的架构参数
        model = UNet(
            T=modelConfig["T"],                           # 扩散步数：前向扩散过程的总步数
            ch=modelConfig["channel"],                     # 基础通道数：网络的基础特征通道数
            ch_mult=modelConfig["channel_mult"],           # 通道倍数：各层通道数的倍数
            attn=modelConfig["attn"],                      # 注意力层位置：指定哪些层使用自注意力
            num_res_blocks=modelConfig["num_res_blocks"],  # 残差块数量：每个分辨率层的残差块数
            dropout=0.                                     # Dropout率：推理时设为0，禁用随机失活
        )
        
        # 加载训练好的模型权重
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)  # 将权重加载到模型中
        print("model load weight done.")  # 打印权重加载完成信息
        
        # 设置模型为评估模式
        model.eval()  # 禁用dropout和batchnorm的训练模式行为
        
        # 创建扩散采样器：用于执行反向扩散过程
        sampler = GaussianDiffusionSampler(
            model,                    # 训练好的UNet模型
            modelConfig["beta_1"],   # 噪声调度起始值
            modelConfig["beta_T"],   # 噪声调度结束值
            modelConfig["T"]          # 扩散步数
        ).to(device)  # 将采样器移动到指定设备
        
        # 从标准正态分布采样初始噪声
        # 这是扩散模型生成过程的起点：从纯噪声开始
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32],  # 张量形状：[批次大小, 通道数, 高度, 宽度]
            device=device                                   # 在指定设备上生成噪声
        )
        
        # 保存初始噪声图像（用于对比）
        # 将噪声从[-1,1]范围转换到[0,1]范围，并裁剪到有效像素值范围
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), 
            nrow=modelConfig["nrow"])  # 保存噪声图像，每行显示nrow张图像
        
        # 执行反向扩散过程：从噪声生成图像
        sampledImgs = sampler(noisyImage)  # 使用采样器执行去噪过程
        
        # 将生成的图像从[-1,1]范围转换到[0,1]范围
        sampledImgs = sampledImgs * 0.5 + 0.5  # 线性变换：从[-1,1]映射到[0,1]
        
        # 保存生成的图像
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledImgName"]), 
            nrow=modelConfig["nrow"])  # 保存生成图像，每行显示nrow张图像