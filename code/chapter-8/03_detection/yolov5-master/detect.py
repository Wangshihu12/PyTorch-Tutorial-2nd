# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from tqdm import tqdm

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL - 模型路径或triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam) - 输入源（文件/目录/URL/摄像头）
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path - 数据集配置文件路径
        imgsz=(640, 640),  # inference size (height, width) - 推理尺寸（高度，宽度）
        conf_thres=0.25,  # confidence threshold - 置信度阈值
        iou_thres=0.45,  # NMS IOU threshold - NMS IOU阈值
        max_det=1000,  # maximum detections per image - 每张图像最大检测数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu - 设备选择
        view_img=False,  # show results - 是否显示结果
        save_txt=False,  # save results to *.txt - 是否保存结果到txt文件
        save_conf=False,  # save confidences in --save-txt labels - 是否在txt标签中保存置信度
        save_crop=False,  # save cropped prediction boxes - 是否保存裁剪的预测框
        nosave=False,  # do not save images/videos - 是否不保存图像/视频
        classes=None,  # filter by class: --class 0, or --class 0 2 3 - 按类别过滤
        agnostic_nms=False,  # class-agnostic NMS - 类别无关的NMS
        augment=False,  # augmented inference - 是否使用增强推理
        visualize=False,  # visualize features - 是否可视化特征
        update=False,  # update all models - 是否更新所有模型
        project=ROOT / 'runs/detect',  # save results to project/name - 保存结果的项目路径
        name='exp',  # save results to project/name - 保存结果的名称
        exist_ok=False,  # existing project/name ok, do not increment - 是否允许覆盖现有项目
        line_thickness=3,  # bounding box thickness (pixels) - 边界框线条粗细（像素）
        hide_labels=False,  # hide labels - 是否隐藏标签
        hide_conf=False,  # hide confidences - 是否隐藏置信度
        half=False,  # use FP16 half-precision inference - 是否使用FP16半精度推理
        dnn=False,  # use OpenCV DNN for ONNX inference - 是否使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # video frame-rate stride - 视频帧率步长
):
    """
    YOLOv5目标检测推理主函数
    
    该函数实现了完整的YOLOv5目标检测推理流程，支持多种输入源和输出格式。
    包括图像、视频、摄像头、屏幕截图等多种输入类型的处理。
    
    Args:
        weights: 模型权重文件路径
        source: 输入源（图像/视频/摄像头/URL等）
        data: 数据集配置文件路径
        imgsz: 推理图像尺寸
        conf_thres: 置信度阈值
        iou_thres: NMS IOU阈值
        max_det: 每张图像最大检测数量
        device: 计算设备
        view_img: 是否实时显示结果
        save_txt: 是否保存检测结果到txt文件
        save_conf: 是否在txt中保存置信度
        save_crop: 是否保存裁剪的检测框
        nosave: 是否不保存结果文件
        classes: 指定检测的类别
        agnostic_nms: 是否使用类别无关的NMS
        augment: 是否使用数据增强推理
        visualize: 是否可视化特征图
        update: 是否更新模型
        project: 结果保存项目路径
        name: 结果保存名称
        exist_ok: 是否允许覆盖现有项目
        line_thickness: 边界框线条粗细
        hide_labels: 是否隐藏标签
        hide_conf: 是否隐藏置信度
        half: 是否使用半精度推理
        dnn: 是否使用OpenCV DNN
        vid_stride: 视频帧率步长
    """
    source = str(source)  # 将输入源转换为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images - 确定是否保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断是否为文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断是否为URL
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # 判断是否为摄像头
    screenshot = source.lower().startswith('screen')  # 判断是否为屏幕截图
    if is_url and is_file:
        source = check_file(source)  # download - 如果是URL文件，下载到本地

    # Directories - 创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run - 创建递增的运行目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir - 创建目录

    # Load model - 加载模型
    device = select_device(device)  # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 加载多后端检测模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型步长、类别名称、PyTorch标志
    imgsz = check_img_size(imgsz, s=stride)  # check image size - 检查图像尺寸

    # Dataloader - 数据加载器
    bs = 1  # batch_size - 批次大小
    if webcam:  # 摄像头输入
        view_img = check_imshow(warn=True)  # 检查是否可以显示图像
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载视频流
        bs = len(dataset)  # 批次大小等于数据流数量
    elif screenshot:  # 屏幕截图输入
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # 加载屏幕截图
    else:  # 文件输入
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载图像/视频文件
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频路径和写入器

    # Run inference - 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup - 模型预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # 初始化计数器、窗口列表、性能分析器
    
    # 遍历数据集进行推理
    for path, im, im0s, vid_cap, s in tqdm(dataset, total=len(dataset)):
        # 预处理阶段
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 转换为PyTorch张量并移动到设备
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 - 转换为半精度或单精度
            im /= 255  # 0 - 255 to 0.0 - 1.0 - 归一化到0-1范围
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim - 添加批次维度

        # Inference - 推理阶段
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 可视化路径
            pred = model(im, augment=augment, visualize=visualize)  # 模型推理

        # NMS - 非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 执行NMS

        # Second-stage classifier (optional) - 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions - 处理预测结果
        for i, det in enumerate(pred):  # per image - 处理每张图像
            seen += 1  # 增加已处理图像计数
            if webcam:  # batch_size >= 1 - 摄像头模式
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 获取路径、图像、帧数
                s += f'{i}: '  # 添加批次索引
            else:  # 文件模式
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # 获取路径、图像、帧数

            p = Path(p)  # to Path - 转换为Path对象
            save_path = str(save_dir / p.name)  # im.jpg - 保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt - 标签文件路径
            s += '%gx%g ' % im.shape[2:]  # print string - 添加图像尺寸信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh - 归一化增益
            imc = im0.copy() if save_crop else im0  # for save_crop - 用于保存裁剪框的图像副本
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 创建标注器
            
            if len(det):  # 如果有检测结果
                # Rescale boxes from img_size to im0 size - 将检测框从推理尺寸缩放到原始图像尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results - 打印结果
                for c in det[:, 5].unique():  # 遍历每个类别
                    n = (det[:, 5] == c).sum()  # detections per class - 每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string - 添加到字符串

                # Write results - 写入结果
                for *xyxy, conf, cls in reversed(det):  # 遍历每个检测框
                    if save_txt:  # Write to file - 保存到文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh - 归一化坐标
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format - 标签格式
                        with open(f'{txt_path}.txt', 'a') as f:  # 写入标签文件
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image - 在图像上添加边界框
                        c = int(cls)  # integer class - 整数类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 构建标签
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制边界框和标签
                    if save_crop:  # 保存裁剪框
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)  # 保存单个裁剪框

            # Stream results - 流式显示结果
            im0 = annotator.result()  # 获取标注后的图像
            if view_img:  # 如果显示图像
                if platform.system() == 'Linux' and p not in windows:  # Linux系统且窗口未创建
                    windows.append(p)  # 添加到窗口列表
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux) - 允许窗口调整大小
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小
                cv2.imshow(str(p), im0)  # 显示图像
                cv2.waitKey(1)  # 1 millisecond - 等待1毫秒

            # Save results (image with detections) - 保存结果（带检测框的图像）
            if save_img:  # 如果保存图像
                if dataset.mode == 'image':  # 图像模式
                    cv2.imwrite(save_path, im0)  # 保存图像
                else:  # 'video' or 'stream' - 视频或流模式
                    if vid_path[i] != save_path:  # new video - 新视频
                        vid_path[i] = save_path  # 更新视频路径
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果存在视频写入器
                            vid_writer[i].release()  # release previous video writer - 释放之前的视频写入器
                        if vid_cap:  # video - 视频文件
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
                        else:  # stream - 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 默认帧率和尺寸
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos - 强制使用mp4后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入器
                    vid_writer[i].write(im0)  # 写入帧

        # Print time (inference-only) - 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results - 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image - 每张图像的速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)  # 打印速度信息
    if save_txt or save_img:  # 如果保存了文件
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # 标签文件统计
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 打印保存路径
    if update:  # 如果更新模型
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning) - 更新模型（修复SourceChangeWarning）


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.engine', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydrone.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
