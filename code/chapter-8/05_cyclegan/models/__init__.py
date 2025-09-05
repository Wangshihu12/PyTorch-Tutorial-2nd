"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """
    根据模型名称动态导入并查找对应的模型类
    
    功能描述：
        该函数通过模型名称动态导入对应的模型模块文件，并在该模块中查找
        符合命名规范的模型类。模型类必须是BaseModel的子类，且命名不区分大小写。
    
    参数说明：
        model_name: 模型名称字符串，例如 'cycle_gan'、'pix2pix' 等
    
    返回值说明：
        返回找到的模型类，如果未找到则程序退出
    
    工作流程：
        1. 构建模块文件名：models.[model_name]_model
        2. 动态导入模块
        3. 构建目标类名：[model_name]Model（去除下划线）
        4. 在模块中查找匹配的类
        5. 验证类是否为BaseModel的子类
    """
    # 构建模型模块的文件名
    model_filename = "models." + model_name + "_model"  # 例如：'models.cycle_gan_model'
    
    # 动态导入模型模块
    modellib = importlib.import_module(model_filename)  # 导入models.cycle_gan_model模块
    
    # 初始化模型类变量
    model = None
    
    # 构建目标模型类名：去除下划线并添加'Model'后缀
    target_model_name = model_name.replace('_', '') + 'model'  # 例如：'cycle_gan' -> 'cycleganmodel'
    
    # 遍历模块中的所有类，查找匹配的模型类
    for name, cls in modellib.__dict__.items():
        # 检查类名是否匹配（不区分大小写）且是否为BaseModel的子类
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls  # 找到匹配的模型类
            break  # 找到后退出循环
    
    # 如果未找到匹配的模型类，打印错误信息并退出程序
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)  # 程序退出
    
    # 返回找到的模型类
    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """
    根据给定的选项创建模型实例
    
    功能描述：
        这是一个工厂函数，用于根据配置选项动态创建相应的模型实例。
        该函数封装了模型查找和实例化的过程，是训练脚本(train.py)和测试脚本(test.py)
        与模型包之间的主要接口。
    
    参数说明：
        opt: 配置选项对象，包含模型类型、网络参数、训练参数等配置信息
            例如：opt.model = 'cycle_gan' 表示创建CycleGAN模型
    
    返回值说明：
        返回创建的模型实例，该实例继承自BaseModel基类
    
    使用示例：
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    # 根据模型名称查找对应的模型类
    model = find_model_using_name(opt.model)  # 例如：opt.model='cycle_gan' -> 查找CycleGANModel类
    
    # 使用找到的模型类和配置选项创建模型实例
    instance = model(opt)  # 调用模型类的构造函数，传入配置参数
    
    # 打印模型创建成功的确认信息
    print("model [%s] was created" % type(instance).__name__)  # 输出：model [CycleGANModel] was created
    
    # 返回创建的模型实例
    return instance
