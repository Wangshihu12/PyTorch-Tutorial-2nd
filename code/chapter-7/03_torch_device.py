# -*- coding: utf-8 -*-
"""
# @file name  : 03_torch_device.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : torch.device 使用 - 演示PyTorch中CUDA设备信息查询和管理的各种方法
"""
import torch

# =============================== 设备配置 ===============================
# 自动检测并使用可用的设备（GPU或CPU）
# 如果CUDA可用，则使用GPU；否则使用CPU
# 这是PyTorch中设备管理的标准做法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # =============================== CUDA设备信息查询 ===============================
    
    # 1. 获取系统中可用的CUDA设备数量
    # 返回值：int，表示可用的GPU数量
    # 如果没有GPU或CUDA不可用，返回0
    print("device_count: {}".format(torch.cuda.device_count()))

    # 2. 获取当前活跃的CUDA设备索引
    # 返回值：int，当前设备索引（从0开始）
    # 如果没有GPU或CUDA不可用，会抛出异常
    print("current_device: ", torch.cuda.current_device())

    # 3. 获取CUDA设备的计算能力
    # @param device: 设备索引，None表示当前设备
    # 返回值：tuple，包含(major, minor)版本号
    # 例如：(7, 5)表示计算能力7.5
    print(torch.cuda.get_device_capability(device=None))

    # 4. 获取CUDA设备的名称
    # 返回值：str，GPU设备的名称
    # 例如："NVIDIA GeForce RTX 3080"
    print(torch.cuda.get_device_name())

    # 5. 检查CUDA是否可用
    # 返回值：bool，True表示CUDA可用，False表示不可用
    # 这个函数检查：
    # - 是否安装了CUDA
    # - 是否有可用的GPU
    # - CUDA版本是否兼容
    print(torch.cuda.is_available())

    # 6. 获取支持的CUDA架构列表
    # 返回值：list，包含支持的CUDA架构字符串
    # 例如：['sm_35', 'sm_50', 'sm_60', 'sm_70', 'sm_75']
    # 这些架构决定了PyTorch可以编译的CUDA内核
    print(torch.cuda.get_arch_list())

    # 7. 获取指定CUDA设备的详细属性
    # @param device: 设备索引，0表示第一个GPU
    # 返回值：_CudaDeviceProperties对象，包含：
    # - name: 设备名称
    # - major/minor: 计算能力版本
    # - total_memory: 总显存大小
    # - multi_processor_count: 流处理器数量
    print(torch.cuda.get_device_properties(0))

    # 8. 获取CUDA设备的内存信息
    # @param device: 设备索引，None表示当前设备
    # 返回值：tuple，包含(free_memory, total_memory)
    # free_memory: 可用显存（字节）
    # total_memory: 总显存（字节）
    print(torch.cuda.mem_get_info(device=None))

    # 9. 获取CUDA设备的内存使用摘要
    # @param device: 设备索引，None表示当前设备
    # @param abbreviated: bool，False表示详细摘要，True表示简化摘要
    # 返回值：str，内存使用情况的详细报告
    # 包括：已分配内存、缓存内存、空闲内存等
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # 10. 清空CUDA缓存
    # 返回值：None
    # 这个函数会释放PyTorch缓存的内存，但不会释放已分配的张量内存
    # 主要用于：
    # - 释放临时缓存的内存
    # - 在内存不足时清理缓存
    # - 优化内存使用
    print(torch.cuda.empty_cache())

