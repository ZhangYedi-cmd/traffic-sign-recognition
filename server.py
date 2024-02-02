# # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# """
# Run a Flask REST API exposing one or more YOLOv5s models
# """
from concurrent.futures import ThreadPoolExecutor
from utils.plots import colors

import torch
import os
import sys
from flask import Flask, request
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, cv2,
                           increment_path, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

# 设置文件和根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型
model = DetectMultiBackend('./traffic.pt', dnn=False, data='data/traffic.yaml', fp16=False)
device = select_device('')  # 选择设备
model.to(device)

# 创建线程池执行器
executor = ThreadPoolExecutor(max_workers=4)

# 仅处理单张图片
def predict(
        filename,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        save_img=True
):
    save_dir = ROOT / 'results'
    # Load model
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    dataset = LoadImages(filename, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            cv2.imwrite(filename, im0)
            return s

# 定义异步执行的函数
def async_predict(image_path):
    future = executor.submit(predict, image_path)
    return future.result()


def asyncFunc(work, **args):
    future = executor.submit(work, **args)
    return future.result()

# @asyncFunc
# def write_image(path):
#

# 定义处理请求的路由
@app.route('/v1/object-detection/traffic', methods=['POST'])
def handle_request():
    if request.files.get('image'):
        # 保存上传的图片
        im_file = request.files['image']
        image_path = os.path.join('images', im_file.filename)  # 确保 'images' 目录存在
        im_file.save(image_path)
        # 使用线程池异步识别
        result = async_predict(image_path)
        return result

# 定义根路由
@app.route('/')
def index():
    return "<span style='color:red'>I am app 1</span>"

# 启动应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 在开发环境中使用
