import math
import torch
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from mtcnn import MTCNN
import cv2
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt


detector = MTCNN()


def get_landmark(link, mode='else'):
    if mode == 'front':
        img = cv2.imread(link, 0)  # imread默认读取三通道彩色图片，加入参数0表示读灰度图
        # img = cv2.resize(img, (120, 120))
        for i in range(120):
            for j in range(120):
                if img[i, j] == 255:
                    img[i, j] = 1
        img = img * 1.0
        face_mask = gaussian_filter(img, sigma=2)
        face_mask = torch.as_tensor(face_mask, dtype=torch.float32).unsqueeze(0)
        # 返回的就是姿态引导的姿态嵌入图
        return face_mask
    # 读取的是BGR图像
    img = cv2.imread(link)
    # 获取到黑色单通道png模板
    face_mask = np.zeros_like(img[:, :, 0], dtype=np.float32)
    # mtcnn检测
    faces = detector.detect_faces(img)
    for face in faces:
        for x, y in face['keypoints'].values():
            face_mask[y, x] = 1.0

    face_mask = gaussian_filter(face_mask, sigma=2)
    # face_mask = transforms.ToTensor()(Image.fromarray(face_mask))[0].unsqueeze(0)
    face_mask = torch.from_numpy(face_mask).unsqueeze(0)  # 转化为tensor

    return face_mask


def cal_conv_outsize(in_size, kernel_size, stride, padding):
    return (in_size + 2 * padding - kernel_size) // stride + 1


def round_half_up(n):
    return math.floor(n + 0.5)


def cal_conv_pad(in_size, out_size, kernel_size, stride):
    return round_half_up((stride * (out_size - 1) + kernel_size - in_size) / 2)


def cal_deconv_pad(in_size, out_size, kernel_size, stride):
    return round_half_up((stride * (in_size - 1) + kernel_size - out_size) / 2)


def same_padding(size, kernel_size, stride, dilation):
    return ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2


def show_img(tensor):
    img = transforms.ToPILImage(tensor)  # tensor 转PIL图片
    plt.imshow(img, camp='gray')
    plt.show()


def set_requires_grad(module, b):
    for param in module.parameters():
        param.requires_grad = b


def save_model(model_state_dict, epoch, dirname):
    # if type(model).__name__ == nn.DataParallel.__name__:
    #     model = model.module
    torch.save(model_state_dict,
               f'{dirname}/epoch_{epoch}_checkpoint.pth')


def resume_model(model, dict_name, dirname, epoch, strict=True):
    if type(model).__name__ == nn.DataParallel.__name__:
        model = model.module
    path = f'{dirname}/epoch_{epoch}_checkpoint.pth'
    if os.path.exists(path):
        state = torch.load(path)[dict_name]
        model.load_state_dict(state, strict=strict)
    else:
        raise FileNotFoundError(f'Not found {path}')


def segment(inp):  # 输入inp是tensor张量
    # # 128x128
    # eyes = inp[0:len(inp), :, 32:32 + 26, 26:26 + 77]
    # nose = inp[0:len(inp), :, 39:39 + 41, 50:50 + 29]
    # mouth = inp[0:len(inp), :, 78:78 + 23, 44:44 + 41]
    # face = inp[0:len(inp), :, 22:22 + 88, 19:19 + 91]

    # 120x120
    # face：17:17+86, 18:18+86
    # eye：30:30+25, 26:26+69
    # nose：39:39+36, 46:46+27
    # mouth：74:74+22, 41:41+41

    eyes = inp[:, :, 16:16+41, 11:11+101]
    nose = inp[:, :, 39:39+35, 46:46+29]
    mouth = inp[:, :, 74:74+21, 41:41+41]
    face = inp[:, :, 17:17+85, 18:18+85]

    return eyes, nose, mouth, face


