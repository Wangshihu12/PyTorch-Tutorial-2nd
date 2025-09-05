"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# 尝试导入wandb库，用于实验跟踪和可视化
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    """
    主程序入口：CycleGAN模型推理脚本
    功能：加载训练好的模型，对测试图像进行推理，并保存结果到HTML文件
    """
    
    # 解析测试参数配置
    opt = TestOptions().parse()  # 获取测试选项配置，包括模型路径、结果保存路径等
    
    # 为测试硬编码一些参数设置
    opt.num_threads = 0   # 测试代码只支持单线程（num_threads = 0）
    opt.batch_size = 1    # 测试代码只支持批大小为1
    opt.serial_batches = True  # 禁用数据打乱；如果需要随机选择图像的结果，请注释此行
    opt.no_flip = True    # 不进行图像翻转；如果需要翻转图像的结果，请注释此行
    opt.display_id = -1   # 不使用visdom显示；测试代码将结果保存到HTML文件
    
    # 创建数据集和模型
    dataset = create_dataset(opt)  # 根据opt.dataset_mode和其他选项创建测试数据集
    model = create_model(opt)      # 根据opt.model和其他选项创建模型
    model.setup(opt)               # 常规设置：加载和打印网络结构；创建调度器

    # 初始化wandb日志记录器（如果启用）
    if opt.use_wandb:
        # 初始化wandb运行，用于实验跟踪和结果可视化
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')  # 为运行添加标签

    # 创建结果保存的网站目录
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录路径
    if opt.load_iter > 0:  # 如果指定了特定的迭代次数（默认为0）
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)  # 在目录名中添加迭代次数
    print('creating web directory', web_dir)
    # 创建HTML网页对象，用于保存结果
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # 设置模型为评估模式
    # 这只会影响像batchnorm和dropout这样的层
    # 对于[pix2pix]：原始pix2pix使用batchnorm和dropout，可以尝试使用和不使用eval()模式
    # 对于[CycleGAN]：不应该影响CycleGAN，因为CycleGAN使用instancenorm且没有dropout
    if opt.eval:
        model.eval()  # 将模型设置为评估模式，禁用dropout等训练时的随机性
    
    # 开始推理循环：遍历测试数据集
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 只对opt.num_test张图像应用模型
            break
            
        # 模型推理步骤
        model.set_input(data)  # 从数据加载器中解包数据
        model.test()           # 运行推理，生成转换后的图像
        visuals = model.get_current_visuals()  # 获取图像结果（包括输入、输出、真实图像等）
        img_path = model.get_image_paths()     # 获取当前处理的图像路径
        
        # 每5张图像打印一次处理进度
        if i % 5 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        # 保存图像结果到HTML网页
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    webpage.save()  # 保存HTML文件，完成所有结果的保存

"""
monet2photo 预训练权重 链接：https://pan.baidu.com/s/1bEPNBbAeqMumpM2pqKwb4w  提取码：q159
"""