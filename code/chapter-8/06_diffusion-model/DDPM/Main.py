from Diffusion.Train import train, eval


def main(model_config=None):
    """
    DDPM模型主函数：配置模型参数并执行训练或推理
    
    功能描述：
        该函数是DDPM（去噪扩散概率模型）的主入口函数，负责配置模型参数
        并根据状态选择执行训练或推理任务。
    
    参数说明：
        model_config: 可选的模型配置字典，如果提供则覆盖默认配置
    
    返回值说明：
        无返回值，直接执行训练或推理任务
    """
    # 默认模型配置字典，包含DDPM模型的所有超参数和设置
    modelConfig = {
        # 运行状态设置
        "state": "eval",  # 运行状态：'train'表示训练模式，'eval'表示推理模式
        
        # 训练相关参数
        "epoch": 200,                    # 训练轮数：总共训练200个epoch
        "batch_size": 80,                # 批大小：每批处理80张图像（适合12G显存的GPU）
        "lr": 1e-4,                      # 学习率：0.0001，用于优化器更新参数
        "grad_clip": 1.,                 # 梯度裁剪：限制梯度范数不超过1.0，防止梯度爆炸
        
        # 扩散过程参数
        "T": 1000,                       # 扩散步数：前向扩散过程的总步数
        "beta_1": 1e-4,                  # 噪声调度起始值：扩散过程开始时的噪声水平
        "beta_T": 0.02,                  # 噪声调度结束值：扩散过程结束时的噪声水平
        
        # 网络架构参数
        "channel": 128,                  # 基础通道数：UNet网络的基础特征通道数
        "channel_mult": [1, 2, 3, 4],    # 通道倍数：各层通道数的倍数，[128, 256, 384, 512]
        "attn": [2],                     # 注意力层位置：在第2层添加自注意力机制
        "num_res_blocks": 2,             # 残差块数量：每个分辨率层使用2个残差块
        "dropout": 0.15,                 # Dropout率：15%的神经元随机失活，防止过拟合
        "multiplier": 2.,                # 乘数因子：用于调整网络容量
        
        # 图像和数据集参数
        "img_size": 32,                  # 图像尺寸：输入图像的大小为32x32像素
        "cifar10_dir": './cifar10',      # CIFAR-10数据集路径：存储训练数据的目录
        
        # 设备设置
        "device": "cuda",                # 计算设备：使用CUDA GPU进行加速计算
        
        # 模型保存和加载路径
        "training_load_weight": None,    # 训练时加载的权重：None表示从头开始训练
        "save_weight_dir": "./Checkpoints/",  # 权重保存目录：训练过程中保存模型检查点
        "test_load_weight": "ckpt_last_.pt",   # 推理时加载的权重：用于生成图像的模型文件
        
        # 生成图像保存设置
        "sampled_dir": "./SampledImgs/",        # 生成图像保存目录
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",  # 带噪声图像文件名
        "sampledImgName": "SampledNoGuidenceImgs.png",     # 生成图像文件名
        "nrow": 8,                       # 图像网格行数：生成图像时每行显示8张图像
    }
    
    # 如果提供了自定义配置，则使用自定义配置覆盖默认配置
    if model_config is not None:
        modelConfig = model_config
    
    # 根据状态选择执行训练或推理任务
    if modelConfig["state"] == "train":
        # 训练模式：调用训练函数，使用配置参数训练DDPM模型
        train(modelConfig)
    else:
        # 推理模式：调用推理函数，使用训练好的模型生成新图像
        eval(modelConfig)


if __name__ == '__main__':
    # 注意修改：
    # 1. state: 训练和推理状态
    # 2. cifar10_dir，根据自己的情况设置文件夹路径
    # 3. batch_size， 12G的1080ti，可以设置80
    main()
