import sys
import time

import mss
import cv2
import pydirectinput
import torch
import win32gui
import win32con
import numpy as np

from tool import FOV
import hub_mouse as ghub
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords, xyxy2xywh, check_img_size, non_max_suppression
from utils.torch_utils import select_device
import pyautogui
import tkinter as tk


def pydirectinput_move(x, y):
    pydirectinput.moveTo(x, y)
    pydirectinput.press('v')


#  time.sleep(1)
# pydirectinput.press('s')

def ghub_move(x, y):
    ghub.mouse_xy(x, y)
    ghub.mouse_down(1)
    ghub.mouse_up(1)
    '''
    FOV_H = FOV(game_width=1920, game_fov=80.0, game_pixel=4538,
                degrees=360, measure_seen=2.5, game_seen=2.5)
    FOV_V = FOV(game_width=1080, game_fov=50.534016, game_pixel=1920,
                degrees=180, measure_seen=2.5, game_seen=2.5)
    '''


def screenshot_mss(img_size, stride, dev, model, monitor):
    with mss.mss() as sct:
        img_0 = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
        img_ = letterbox(img_0, img_size, stride=stride, auto=True)[0]
        img_ = img_.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_ = np.ascontiguousarray(img_)
        img_ = torch.from_numpy(img_).to(dev)
        img_ = img_.half() if model.fp16 else img_.float()  # uint8 to fp16/32
        img_ /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img_.shape) == 3:
        img_ = img_[None]
    return img_, img_0


def screenshot_win32():
    pass


def getMonitor(screenW, screenH, Width, Height) -> dict:
    """
    返回monitor字典
    """

    LeftTopX, LeftTopY = int(screenW / 2 - Width / 2), int(screenH / 2 - Height / 2)
    return {'left': LeftTopX, 'top': LeftTopY, 'width': Width, 'height': Height}


def show_img(WindowName, img, xywh):
    for i in xywh:
        cv2.rectangle(img, [i[0], i[1]], [i[2], i[3]], (0, 255, 0), thickness=3)

    cv2.imshow(WindowName, img)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        sys.exit(0)


def argument():
    dt = {}

    def add(**kwargs):
        for k, v in kwargs.items():
            dt[k] = v
        dt['screenshot'] = screenshot_mss if dt['screenshot'] == 'mss' else screenshot_win32
        if len(dt['aiming_range']) == 4:
            dt['aiming_range'] = {'left': dt['aiming_range'][0],
                                  'top': dt['aiming_range'][1],
                                  'width': dt['aiming_range'][2],
                                  'height': dt['aiming_range'][3]}
        else:
            dt['aiming_range'] = getMonitor(*get_resolution(), *dt['aiming_range']) if dt['aiming_range'] \
                else getMonitor(*get_resolution() * 2)
        dt['move_control'] = pydirectinput_move
        return dt

    return add


def get_resolution(w=-1, h=-1):
    if w == -1 and h == -1:
        root = tk.Tk()
        return root.winfo_screenwidth(), root.winfo_screenheight()
    else:
        return w, h


def run(opt):
    monitor = opt['aiming_range']
    model = DetectMultiBackend(opt['weights'], device=opt['device'], dnn=False, data=opt['data'], fp16=False)  # 加载模型
    stride = model.stride
    imgsz = check_img_size(opt['imgsz'], s=stride)
    if opt['view_show']:
        cv2.namedWindow('lol', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 显示窗口可拖拽大小
        cv2.setWindowProperty("lol", cv2.WND_PROP_TOPMOST, 1)  # 窗口置顶
    while True:
        img, img0 = screenshot_mss(imgsz, stride, opt['device'], model, monitor)
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred,
                                   opt['conf_thres'],
                                   opt['iou_thres'],
                                   opt['classes'],
                                   opt['agnostic_nms'],
                                   max_det=opt['max_det'])
        xywh = []

        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh_s = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    x_center, y_center, width, height = xywh_s[0], xywh_s[1], xywh_s[2], xywh_s[3]
                    x_center, width = gn[0] * float(x_center), gn[0] * float(width)
                    y_center, height = gn[1] * float(y_center), gn[1] * float(height)
                    xywh.append([int(x_center - width / 2.), int(y_center - height / 2.), int(x_center + width / 2.),
                                 int(y_center + height / 2.)])

            if xywh:
                x, y, w, h = xywh[0][0], xywh[0][1], xywh[0][2] - xywh[0][0], xywh[0][3] - xywh[0][1]
                # 左上xy，长高wh
                centerX = x + monitor["left"] - 10
                centerY = round(h / 2) + y + monitor["top"]
            #    pydirectinput_move(centerX, centerY) # 键鼠控制会严重拖慢识别速度，后续要开子线程专门负责键鼠控制
            else:
                pass
        if opt['view_show']:
            show_img('lol', img0, xywh)  # 显示窗口


if __name__ == '__main__':
    # yolo v5-6.0
    parser = argument()
    run(parser(
        screenshot='mss',  # 屏幕捕获方式 有 mss和win32 两种 mss最快,但不支持指定窗口,win32稍慢，但支持指定窗口
        aiming_range=(1, 1, 1048, 1070),  # 检测范围  方式1:以屏幕为中心的方形(长w,高h)  方式2:(左上X,左上Y,长w,高h) 不填为全屏
        #  program_capture='',  # win32指定的窗口，不填为全屏
        #  move_control=1,  # 控制方式 1=pydirectinput库 需要管理员权限打开, 2=罗技驱动,暂时有问题
        view_show=True,  # 是否显示实时检测窗口
        weights=r'F:\yolov5-master\runs\train\exp3\weights\best.pt',  # 权重(模型)-所在位置
        data=r'F:\yolov5-master\data\data\coco128.yaml',  # 模型所对应的名字 -所在位置
        conf_thres=0.8,  # 精度
        iou_thres=0.9,  # 交并比(重合的"框"是否合并为一个)
        classes=0,  # 选择目标(对应yaml文件标签,None全选)
        agnostic_nms=False,  # 增强检测，可能会出错 类似精度
        max_det=10,  # 每张图最多识别数量
        imgsz=640,  # 与权重对应
        resolution=get_resolution(),  # 桌面分辨率,不填为自动获取 自瞄范围 方式1 根据此计算
        device=select_device(''),  # 自动选择Gpu,cpu 优先选择GPU 默认不需要动,除非要手动选择设备
    ))
