"""
# ==============================
# tartanTrajFlowDataset.py
# library for DytanVO data I/O
# Author: Wenshan Wang, Shihao Shen
# Date: 3rd Jan 2023
# ==============================
"""
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from evaluator.transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

# 继承Pytorch的dataset类
class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        # 获取指定目录下的所有文件和子目录的名称列表
        files = listdir(imgfolder)
        # 将文件夹下所有png和jpg结尾的文件路径，储存在self.rgbfiles列表中
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        # sort() 方法会按照字符串的字典序（lexicographic order）对列表进行升序排序。
        # 字典序是一种字符排序规则，它比较两个字符串的每个字符，并按照字符的ASCII码值进行排序
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    # __getitem__()允许对象能够像列表或字典一样使用索引操作来访问元素：obj[index]
    def __getitem__(self, idx):
        # strip()字符串方法，用于去除字符串两端的空白字符
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        # make_intrinsics_layer()生成内参矩阵层，用于将图像中的像素坐标转换为相机坐标系中的归一化坐标。
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        # 将内参数矩阵层添加到字典中去
        res['intrinsic'] = intrinsicLayer 
        # 如果某些数据集的图片是需要transform的，那么对照片和内参矩阵层都进行transform
        if self.transform:
            res = self.transform(res)

        res['img1_raw'] = img1
        res['img2_raw'] = img2

        return res