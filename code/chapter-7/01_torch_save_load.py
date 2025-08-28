# -*- coding:utf-8 -*-
"""
@file name  : 01_torch_save_load.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-022
@brief      : 模型保存与加载
"""
import torch
import torchvision.models as models
from torchinfo import summary


if __name__ == "__main__":
    # =============================== 模型保存演示（torch.save） ===============================
    # 设置保存路径：模型状态字典文件
    path_state_dict = "resnet50_state_dict_2022.pth"
    
    # 创建ResNet50模型实例（不使用预训练权重）
    resnet_50 = models.resnet50(pretrained=False)

    # 模拟训练过程：将模型参数修改为特定值
    # 这里只是为了演示，实际训练中参数会通过梯度下降更新
    print("训练前: ", resnet_50.conv1.weight[0, ...])  # 显示第一个卷积层第一个卷积核的权重
    
    # 将所有模型参数设置为2022（模拟训练后的参数值）
    for p in resnet_50.parameters():
        p.data.fill_(2022)  # fill_方法会直接修改张量的值
    
    print("训练后: ", resnet_50.conv1.weight[0, ...])  # 显示修改后的权重

    # 保存模型状态字典到文件
    net_state_dict = resnet_50.state_dict()  # 获取模型的所有参数和缓冲区
    torch.save(net_state_dict, path_state_dict)  # 将状态字典保存到指定路径
    
    # =============================== 模型加载演示（torch.load） ===============================
    # 创建一个新的ResNet50模型实例（用于演示加载）
    resnet_50_new = models.resnet50(pretrained=False)

    # 显示新模型初始化后的权重（应该是随机值）
    print("初始化: ", resnet_50_new.conv1.weight[0, ...])
    
    # 从文件加载保存的状态字典
    state_dict = torch.load(path_state_dict)
    
    # 将加载的状态字典应用到新模型上
    resnet_50_new.load_state_dict(state_dict)
    
    # 显示加载后的权重（应该与之前保存的2022值相同）
    print("加载后: ", resnet_50_new.conv1.weight[0, ...])

    # =============================== torchvision官方脚本示例 ===============================
    # 参考链接：https://github.com/pytorch/vision/blob/fa347eb9f38c1759b73677a11b17335191e3f602/references/classification/train.py
    
    # 创建完整的checkpoint字典，包含训练所需的所有信息
    # checkpoint = {
    #     "model": model_without_ddp.state_dict(),      # 模型参数（去除分布式训练包装器）
    #     "optimizer": optimizer.state_dict(),          # 优化器状态（包括动量、学习率等）
    #     "lr_scheduler": lr_scheduler.state_dict(),    # 学习率调度器状态
    #     "epoch": epoch,                               # 当前训练轮数
    # }
    
    # 设置保存路径，包含epoch信息便于管理
    # path_save = "model_{}.pth".format(epoch)
    
    # 保存完整的checkpoint到文件
    # torch.save(checkpoint, path_save)
    
    # =============================== 训练恢复（resume）演示 ===============================
    # 从checkpoint恢复训练状态
    
    # 加载checkpoint文件
    # checkpoint = torch.load(path_save, map_location="cpu")  # map_location="cpu"确保在CPU上加载
    
    # 恢复模型参数
    # model.load_state_dict(checkpoint["model"])
    
    # 恢复优化器状态（包括动量、学习率等）
    # optimizer.load_state_dict(checkpoint["optimizer"])
    
    # 恢复学习率调度器状态
    # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    # 设置下一个epoch的起始值
    # start_epoch = checkpoint["epoch"] + 1







