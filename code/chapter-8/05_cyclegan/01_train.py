"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    """
    主程序入口：CycleGAN模型训练脚本
    功能：执行完整的训练流程，包括数据加载、模型训练、可视化输出和模型保存
    """

    import time
    print('sleep')

    # 解析训练参数配置
    opt = TrainOptions().parse()   # 获取训练选项配置，包括数据集路径、模型类型、超参数等
    # 创建数据集加载器
    dataloader = create_dataset(opt)  # 根据opt.dataset_mode和其他选项创建数据集
    dataset_size = len(dataloader)    # 获取数据集中图像的总数量
    print('训练图像数据 = %d' % dataset_size)

    # 创建和初始化模型
    model = create_model(opt)      # 根据opt.model和其他选项创建模型（如CycleGAN、pix2pix等）
    model.setup(opt)               # 常规设置：加载和打印网络结构；创建学习率调度器
    visualizer = Visualizer(opt)   # 创建可视化器，用于显示/保存图像和绘制损失曲线
    total_iters = 0                # 训练的总迭代次数计数器

    # 开始训练循环：外层循环遍历不同的epoch
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        # 外层循环：遍历不同的训练轮次；模型保存频率为<epoch_count>和<epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # 记录整个epoch的开始时间
        iter_data_time = time.time()    # 记录每次迭代数据加载的时间
        epoch_iter = 0                  # 当前epoch内的训练迭代次数，每个epoch重置为0
        visualizer.reset()              # 重置可视化器：确保每个epoch至少保存一次结果到HTML
        model.update_learning_rate()   # 在每个epoch开始时更新学习率
        
        # 内层循环：遍历当前epoch内的所有数据批次
        for i, data in enumerate(dataloader):  
            # 内层循环：在一个epoch内遍历所有数据批次
            iter_start_time = time.time()  # 记录每次迭代计算开始的时间
            
            # 计算数据加载时间（仅在需要打印时计算）
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time  # 数据加载耗时

            # 更新迭代计数器
            total_iters += opt.batch_size    # 总迭代次数增加batch_size
            epoch_iter += opt.batch_size     # 当前epoch迭代次数增加batch_size
            
            # 模型训练步骤
            model.set_input(data)         # 从数据集中解包数据并应用预处理
            model.optimize_parameters()   # 计算损失函数、获取梯度、更新网络权重

            # 可视化显示：在visdom上显示图像并保存到HTML文件
            if total_iters % opt.display_freq == 0:   
                # 每隔display_freq次迭代显示图像
                save_result = total_iters % opt.update_html_freq == 0  # 判断是否需要保存结果到HTML
                model.compute_visuals()  # 计算可视化结果
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印训练损失：打印训练损失并保存日志信息到磁盘
            if total_iters % opt.print_freq == 0:    
                # 每隔print_freq次迭代打印损失信息
                losses = model.get_current_losses()  # 获取当前损失值
                t_comp = (time.time() - iter_start_time) / opt.batch_size  # 计算每次迭代的平均计算时间
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)  # 打印当前损失
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)  # 绘制损失曲线

            # 保存最新模型：每隔save_latest_freq次迭代缓存最新模型
            if total_iters % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'  # 根据配置决定保存后缀
                model.save_networks(save_suffix)  # 保存网络权重

            iter_data_time = time.time()  # 更新数据加载时间记录点
            
        # 按epoch保存模型：每隔save_epoch_freq个epoch缓存模型
        if epoch % opt.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')  # 保存最新模型
            model.save_networks(epoch)     # 保存当前epoch的模型

        # 打印epoch结束信息
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
