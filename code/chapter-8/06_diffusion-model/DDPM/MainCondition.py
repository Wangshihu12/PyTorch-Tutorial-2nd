from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    """
    条件DDPM模型主函数：配置条件扩散模型参数并执行训练或推理
    
    功能描述：
        该函数是条件DDPM（Conditional Denoising Diffusion Probabilistic Models）的主入口函数，
        支持基于类别标签的条件生成。相比无条件的DDPM，条件DDPM可以根据指定的类别生成
        特定类型的图像，提高生成的可控性和质量。
    
    参数说明：
        model_config: 可选的模型配置字典，如果提供则覆盖默认配置
    
    返回值说明：
        无返回值，直接执行训练或推理任务
    """
    # 条件DDPM模型配置字典，包含条件生成的所有超参数和设置
    modelConfig = {
        # 运行状态设置
        "state": "train",  # 运行状态：'train'表示训练模式，'eval'表示推理模式
        
        # 训练相关参数
        "epoch": 70,                    # 训练轮数：总共训练70个epoch（比无条件DDPM少，因为条件信息有助于收敛）
        "batch_size": 80,                # 批大小：每批处理80张图像
        "lr": 1e-4,                      # 学习率：0.0001，用于优化器更新参数
        "grad_clip": 1.,                 # 梯度裁剪：限制梯度范数不超过1.0，防止梯度爆炸
        
        # 扩散过程参数（相比无条件DDPM有所调整）
        "T": 500,                        # 扩散步数：500步前向扩散过程（比无条件DDPM的1000步少）
        "beta_1": 1e-4,                  # 噪声调度起始值：扩散过程开始时的噪声水平
        "beta_T": 0.028,                 # 噪声调度结束值：0.028（比无条件DDPM的0.02稍高）
        
        # 网络架构参数
        "channel": 128,                  # 基础通道数：UNet网络的基础特征通道数
        "channel_mult": [1, 2, 2, 2],    # 通道倍数：各层通道数的倍数，[128, 256, 256, 256]
        "num_res_blocks": 2,             # 残差块数量：每个分辨率层使用2个残差块
        "dropout": 0.15,                 # Dropout率：15%的神经元随机失活，防止过拟合
        "multiplier": 2.5,               # 乘数因子：2.5（比无条件DDPM的2.0稍高，增加网络容量）
        
        # 图像参数
        "img_size": 32,                  # 图像尺寸：输入图像的大小为32x32像素
        
        # 设备设置
        "device": "cuda:0",              # 计算设备：使用第一个CUDA GPU进行加速计算
        
        # 条件生成相关参数
        "w": 1.8,                        # 引导权重：控制条件信息的强度，值越大条件引导越强
        
        # 模型保存和加载路径
        "save_dir": "./CheckpointsCondition/",  # 条件模型权重保存目录
        "training_load_weight": None,    # 训练时加载的权重：None表示从头开始训练
        "test_load_weight": "ckpt_63_.pt",   # 推理时加载的权重：使用第63个epoch的检查点
        
        # 生成图像保存设置
        "sampled_dir": "./SampledImgs/",        # 生成图像保存目录
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",  # 带噪声的条件引导图像文件名
        "sampledImgName": "SampledGuidenceImgs.png",     # 条件引导生成图像文件名
        "nrow": 8,                       # 图像网格行数：生成图像时每行显示8张图像
    }
    
    # 如果提供了自定义配置，则使用自定义配置覆盖默认配置
    if model_config is not None:
        modelConfig = model_config
    
    # 根据状态选择执行训练或推理任务
    if modelConfig["state"] == "train":
        # 训练模式：调用条件训练函数，使用配置参数训练条件DDPM模型
        train(modelConfig)
    else:
        # 推理模式：调用条件推理函数，使用训练好的条件模型生成指定类别的图像
        eval(modelConfig)


if __name__ == '__main__':
    main()
