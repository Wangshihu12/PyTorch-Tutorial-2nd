import time
import torch
import numpy as np
from torchvision import models

# 开启 TF32 高性能模式
torch.set_float32_matmul_precision('high')  # 启用 TF32

# 定义torch.compile的三种编译模式列表
# - default: 默认模式，平衡编译时间和运行性能
# - reduce-overhead: 减少开销模式，优先减少编译时间
# - max-autotune: 最大自动调优模式，优先提升运行性能
mode_list = "default reduce-overhead max-autotune".split()

# 设置计算设备：如果CUDA可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个简单的三角函数，用于测试torch.compile的性能提升
def sin_func(x):
    return torch.sin(x) + torch.cos(x)

# 设置运行次数，用于统计性能数据
run_times = 100000
# 创建输入数据：一个值为1的张量，并移动到指定设备（GPU或CPU）
i_data = torch.tensor(1).to(device)

print("实验一: 测试三角函数使用torch.compile的性能提升")
# 遍历不同的编译模式进行性能测试
for mode in mode_list:
    # 确保GPU操作完成，用于精确计时
    torch.cuda.synchronize()
    time_0 = time.time()
    
    # 使用torch.compile编译函数，指定编译模式
    module_compiled = torch.compile(sin_func, mode=mode)
    
    # 再次同步，确保编译完成
    torch.cuda.synchronize()
    time_1 = time.time()
    
    # 预热阶段：运行原始函数和编译后的函数各一次
    # 这很重要，因为第一次运行可能包含初始化开销
    sin_func(i_data)
    module_compiled(i_data)
    
    # 测试原始函数性能：记录开始时间
    torch.cuda.synchronize()
    time_2 = time.time()
    
    # 运行原始函数多次，统计总耗时
    for i in range(run_times):
        sin_func(i_data)
        
    # 记录原始函数运行完成时间
    torch.cuda.synchronize()
    time_3 = time.time()
    
    # 测试编译后函数性能：运行编译后的函数多次
    for i in range(run_times):
        module_compiled(i_data)
    
    # 记录编译后函数运行完成时间
    torch.cuda.synchronize()
    time_4 = time.time()
    
    # 计算各项时间指标
    compile_time = time_1 - time_0      # 编译耗时
    pre_time = time_3 - time_2          # 编译前运行耗时
    post_time = time_4 - time_3         # 编译后运行耗时
    speedup_ratio = (pre_time - post_time)/pre_time  # 性能提升比例
    
    # 输出性能测试结果
    print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")
    

# 实验二：测试ResNet18模型使用torch.compile的性能提升

# 加载预训练的ResNet18模型并移动到指定设备
resnet18 = models.resnet18().to(device)
# 设置为评估模式，关闭dropout和batch normalization的随机性
resnet18.eval()
# 创建假输入数据：16张图片，每张3通道，尺寸224x224
fake_img = torch.randn(16, 3, 224, 224).to(device)

# 设置运行次数，由于模型较大，减少运行次数以节省时间
run_times = 100

print("实验二: 测试ResNet18模型使用torch.compile的性能提升")
# 使用torch.no_grad()上下文管理器，关闭梯度计算以节省内存和提升速度
with torch.no_grad():
    # 遍历不同的编译模式进行性能测试
    for mode in mode_list:
        # 确保GPU操作完成，用于精确计时
        torch.cuda.synchronize()
        time_0 = time.time()
        
        # 使用torch.compile编译ResNet18模型，指定编译模式
        module_compiled = torch.compile(resnet18, mode=mode)
        
        # 再次同步，确保编译完成
        torch.cuda.synchronize()
        time_1 = time.time()
        
        # 预热阶段：运行原始模型和编译后的模型各一次
        # 对于大型模型，预热非常重要，可以避免首次运行的初始化开销
        resnet18(fake_img)
        module_compiled(fake_img)
        
        # 测试原始模型性能：记录开始时间
        torch.cuda.synchronize()
        time_2 = time.time()
        
        # 运行原始模型多次，统计总耗时
        for i in range(run_times):
            resnet18(fake_img)
        
        # 记录原始模型运行完成时间
        torch.cuda.synchronize()
        time_3 = time.time()
        
        # 测试编译后模型性能：运行编译后的模型多次
        for i in range(run_times):
            module_compiled(fake_img)
        
        # 记录编译后模型运行完成时间
        torch.cuda.synchronize()
        time_4 = time.time()

        # 计算各项时间指标
        compile_time = time_1 - time_0      # 编译耗时
        pre_time = time_3 - time_2          # 编译前运行耗时
        post_time = time_4 - time_3         # 编译后运行耗时
        speedup_ratio = (pre_time - post_time)/pre_time  # 性能提升比例

        # 输出ResNet18模型的性能测试结果
        print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")

# 实验三：BERT

from transformers import BertModel, BertTokenizer
import time

# bert = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # 准备一批输入数据
# input_text = "Here is some text to encode"
# inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
# inputs = {k: v.to(device) for k, v in inputs.items()}
# bert.to(device)
# bert.eval()

# run_times = 100
# with torch.no_grad():
#     for mode in mode_list:
        
#         # 编译
#         torch.cuda.synchronize()
#         time_0 = time.time()
#         bert_compiled = torch.compile(bert, mode=mode)
#         torch.cuda.synchronize()
#         time_1 = time.time()
        
#         # warmup 非常关键！
#         bert(**inputs)
#         bert_compiled(**inputs)

#         torch.cuda.synchronize()
#         time_2= time.time()
#         for _ in range(run_times): 
#             _ = bert(**inputs)

#         torch.cuda.synchronize()
#         time_3= time.time()
#         for _ in range(run_times):
#             _ = bert_compiled(**inputs)
        
#         torch.cuda.synchronize()
#         time_4= time.time()
        
#         compile_time = time_1 - time_0
#         pre_time = time_3 - time_2
#         post_time = time_4 - time_3
#         speedup_ratio = (pre_time - post_time)/pre_time
        
        
#         print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")



# 实验四：测试NumPy函数使用torch.compile的性能提升

# 设置运行次数，用于统计性能数据
run_times = 100

# 定义一个简单的NumPy函数：计算两个数组的张量积并求和
def numpy_fn2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

# 定义一个复杂的NumPy函数：包含数据预处理、张量运算和后处理的完整流程
def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Step 1: 对输入数组进行标准化，使其均值为0，方差为1
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    
    # 避免除零错误：如果标准差为0，则设为1
    X_std[X_std == 0] = 1
    Y_std[Y_std == 0] = 1
    
    # 执行标准化操作
    X_normalized = (X - X_mean) / X_std
    Y_normalized = (Y - Y_mean) / Y_std
    
    # Step 2: 执行张量积运算，然后对最后两个维度求和
    # 使用广播机制实现高效的张量运算
    intermediate_result = np.sum(X_normalized[:, :, None] * Y_normalized[:, None, :], axis=(-2, -1))
    
    # Step 3: 应用阈值裁剪，将值限制在[-1, 1]范围内
    intermediate_result = np.clip(intermediate_result, -1, 1)
    
    # Step 4: 应用指数函数增加非线性特性
    result = np.exp(intermediate_result)
    
    # Step 5: 添加正则化项以防止过拟合
    regularization_term = 0.001 * np.sum(X_normalized ** 2 + Y_normalized ** 2, axis=1)
    result += regularization_term
    
    return result

# 创建测试数据：两个随机生成的数组，用于性能测试
x = np.random.randn(1024, 64)  # 1024个样本，每个64维特征
y = np.random.randn(1024, 64)  # 1024个样本，每个64维特征

print("实验四: 测试NumPy函数使用torch.compile的性能提升")
# 遍历不同的编译模式进行性能测试
for mode in mode_list:
    # 确保GPU操作完成，用于精确计时
    torch.cuda.synchronize()
    time_0 = time.time()
    
    # 使用torch.compile编译NumPy函数，指定编译模式
    numpy_fn_compiled = torch.compile(numpy_fn, mode=mode)
    
    # 再次同步，确保编译完成
    torch.cuda.synchronize()
    time_1 = time.time()

    # 预热阶段：运行原始函数和编译后的函数各一次
    # 对于复杂的NumPy函数，预热非常重要，可以避免首次运行的初始化开销
    numpy_fn(x, y)
    numpy_fn_compiled(x, y)

    # 测试原始函数性能：记录开始时间
    torch.cuda.synchronize()
    time_2 = time.time()
    
    # 运行原始函数多次，统计总耗时
    for i in range(run_times):
        numpy_fn(x, y)

    # 记录原始函数运行完成时间
    torch.cuda.synchronize()
    time_3 = time.time()
    
    # 测试编译后函数性能：运行编译后的函数多次
    for i in range(run_times):
        numpy_fn_compiled(x, y)

    # 记录编译后函数运行完成时间
    torch.cuda.synchronize()
    time_4 = time.time()

    # 计算各项时间指标
    compile_time = time_1 - time_0      # 编译耗时
    pre_time = time_3 - time_2          # 编译前运行耗时
    post_time = time_4 - time_3         # 编译后运行耗时
    speedup_ratio = (pre_time - post_time)/pre_time  # 性能提升比例

    # 输出NumPy函数的性能测试结果
    print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")

