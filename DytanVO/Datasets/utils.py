"""
# ==============================
# utils.py
# misc library for DytanVO
# Author: Wenshan Wang, Shihao Shen
# Date: 3rd Jan 2023
# ==============================
"""

from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import _colormaps as colormaps 
import os
from scipy.spatial.transform import Rotation as R

if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

import time
# ===== general functions =====
# 在Python中，object 是所有类的基类（base class）。它是Python中的根类，其他所有类都直接或间接地继承自object。
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    # 构造函数
    def __init__(self, transforms):
        self.transforms = transforms

    # 让类实例像函数一样被调用
    # test = Compose(List[Transform])
    # result = test(img2)    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale!=1 and 'flow' in sample :
            sample['flow'] = cv2.resize(sample['flow'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'intrinsic' in sample :
            sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'fmask' in sample :
            sample['fmask'] = cv2.resize(sample['fmask'],
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        return sample

class CropCenter(object):
    """Crops a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        hh, ww = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if ww == tw and hh == th:
            return sample

        # resize the image if the image size is smaller than the target size
        scale_h = max(1, float(th)/hh)
        scale_w = max(1, float(tw)/ww)
        
        if scale_h>1 or scale_w>1:
            w = int(round(ww * scale_w)) # w after resize
            h = int(round(hh * scale_h)) # h after resize
        else:
            w, h = ww, hh

        if scale_h != 1. or scale_w != 1.: # resize the data
            resizedata = ResizeData(size=(h, w))
            sample = resizedata(sample)

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            sample[kk] = img[y1:y1+th,x1:x1+tw,...]

        return sample

class ResizeData(object):
    """Resize the data in a dict
    """

    def __init__(self, size):
        # 如果size是一个数字，那么就将其转为一个元组，表示调整后的高度和宽度相等
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        # 如果size是一个元素，直接使用它来表示目标的大小
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample
        scale_w = float(tw)/w
        scale_h = float(th)/h

        for kk in kks:
            if sample[kk] is None:
                continue
            sample[kk] = cv2.resize(sample[kk], (tw,th), interpolation=cv2.INTER_LINEAR)

        if 'flow' in sample:
            sample['flow'][...,0] = sample['flow'][...,0] * scale_w
            sample['flow'][...,1] = sample['flow'][...,1] * scale_h

        return sample

class ToTensor(object):
    def __call__(self, sample):
        kks = list(sample)

        for kk in kks:
            data = sample[kk]
            data = data.astype(np.float32) 
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)  # add a dummy channel
                
            if len(data.shape) == 3 and data.shape[0]==3: # normalization of rgb images
                data = data/255.0

            sample[kk] = torch.from_numpy(data.copy()) # copy to make memory continuous

        return sample

def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m) 
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

# 计算给定的水平位移（du）和垂直位移（dv）所对应的角度和距离。
def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    # 计算dv和du反正切值
    a = np.arctan2( dv, du )
    # 角度偏移量
    angleShift = np.pi

    # 如果需要转为角度制
    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")
    # 对应位移向量的模
    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

# 光流可视化
# 在光流可视化中，通常将光流的方向表示为色调，光流的大小（或速度）表示为饱和度或明度。
# 通过这样的映射，可以将不同方向和速度的光流表示为不同的颜色和亮度，以便更直观地观察光流场的结构和运动信息。
def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """
    # 计算光流的方向ang，和像素点位移大小mag，因为false所以保存为弧度值
    # flownp光流数组
    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue色调, Saturation饱和度, Value明度 colour model 使用HSV颜色模型
    # 创建一个形状为(高度，宽度，3)的全零数组，3表示hsv三个通道
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    # 创建bool数组am，用于标记ang中小于0的元素
    am = ang < 0
    # 将数组ang中小于0的元素加上2π（2 * π，约等于 6.283185...），以将其调整到 [0, 2π) 的范围内
    ang[am] = ang[am] + np.pi * 2

    # 色调：将方向ang+角度偏移量(pi或180度)，然后除2pi，将方向角度归一化在(0,1)之间
    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    # 饱和度：将像素点的位移也归一化为(0,1)
    hsv[ :, :, 1 ] = mag / maxF * n
    # 明度：也在(0,1)之间
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n
    
    # clip()将数组中小于 0 的值设置为 0，大于 1 的值设置为 1，然后乘上hueMax，用于及控制色调亮度
    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    # 将第2，3个通道也缩放到（0,255）之间
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)
    # 将hsv颜色空间转为rgb颜色空间
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        # 掩码：不等于255的元素为true,等于255的为false
        mask = mask != 255
        # 将BGR图像数组 bgr 中满足条件的像素值（由布尔掩码 mask 确定）设置为黑色。
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr


def dataset_intrinsics(dataset='tartanair', is_15mm=False):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
        baseline = None  # to be determined using load_kitti_intrinsics
    elif dataset == 'airdos':
        focalx, focaly, centerx, centery = 772.54834, 772.54834, 320.0, 180.0
        baseline = 1
    elif dataset == 'rs_d435':
        focalx, focaly, centerx, centery = 384.5080871582031, 384.5080871582031, 316.88897705078125, 240.05723571777344
        baseline = 0.05
    elif dataset == 'sceneflow':
        focalx, focaly, centerx, centery = 1050.0, 1050.0, 479.5, 269.5
        if is_15mm:
            focalx = focaly = 450.0
        baseline = 0.5
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
        baseline = 1
    elif dataset == 'commaai':
        focalx, focaly, centerx, centery = 910.0, 910.0, 582.0, 437.0
        baseline = 1
    else:
        return None
    return focalx, focaly, centerx, centery, baseline



def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    # cm = plt.cm.get_cmap('Spectral')
    cm = colormaps['Spectral']

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

# 生成内参矩阵层，用于将图像中的像素坐标转换为相机坐标系中的归一化坐标。
def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))        # 创建宽高分为别w,h的网格，ww/hh为x/y坐标
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx     # 对坐标进行归一化处理,ox/oy为偏移,fx/fy为焦距
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    # np.stack()将数组在axis轴上进行堆叠，默认axis=0即新的一个维度
    # numpy.transpose(1，2，0)将原来排序为(0,1,2)的数组转为(1,2,0)的排序
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0) 

    return intrinsicLayer

def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    data = {}

    with open(filename, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    P2 = np.reshape(data['P2'], (3,4))
    P3 = np.reshape(data['P3'], (3,4))
    focalx, focaly, centerx, centery = float(P2[0,0]), float(P2[1,1]), float(P2[0,2]), float(P2[1,2])
    baseline = P2[0,3] / P2[0,0] - P3[0,3] / P3[0,0]

    return focalx, focaly, centerx, centery, baseline

# 读取场景流的真值
def load_sceneflow_extrinsics(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 因为场景流是3维空间有深度信息，所以有左右2个相机
    l_exts = []
    r_exts = []
    for l in lines:
        if 'L ' in l:
            l_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
        if 'R ' in l:
            r_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))

    if 'into_future' in filename:
        fids = np.arange(0, len(l_exts))
    else:
        fids = np.arange(len(l_exts) - 1, -1, -1)
    
    # assuming left camera is used by default
    # 创建一个单位矩阵identity matrix
    camT = np.eye(4); camT[1,1] = -1; camT[2,2] = -1  # Sceneflow uses Blender's coordinate system
    pose_quats = []
    pose = np.eye(4)
    for fid in fids:
        ext0 = l_exts[fid]
        ext1 = l_exts[fid+1] if 'into_future' in filename else l_exts[fid-1]
        motion = camT.dot(np.linalg.inv(ext0)).dot(ext1).dot(camT)  # ext is from camera space to world space
        pose = pose @ motion    # @是矩阵乘法运算
        pose_quat = np.zeros(7) # 位姿用四元数(旋转)+3维(平移)坐标来表示
        pose_quat[3:] = R.from_matrix(pose[:3,:3]).as_quat()
        pose_quat[:3] = pose[:3,3]
        pose_quats.append(pose_quat)
    
    return pose_quats