# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Shihao Shen, CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# np.set_printoptions()用于设置打印（显示）NumPy数组时的格式选项。
# precision：设置打印数组时的浮点数精度（小数位数）。默认值为8。
# suppress：设置是否使用科学计数法来打印小数。默认值为False，即不使用科学计数法。如果将其设置为True，则打印小数时会使用固定的小数位数，而不是科学计数法。
# threshold：设置NumPy数组打印时的阈值。当数组的大小超过此阈值时，将省略中间部分，只显示头部和尾部。默认值为1000。
np.set_printoptions(precision=4, suppress=True, threshold=10000)

from torch.autograd import Variable
from Network.VONet import VONet     # 基于TartanVo的视觉里程计的神经网络
from Network.rigidmask.VCNplus import SegNet, WarpModule, flow_reg  # 进行语义分割的网络
from Datasets.utils import CropCenter, ResizeData
from Datasets.cowmask import cow_masks
from evaluator.transformation import se2SE

class DytanVO(object):
    def __init__(self, vo_model_name, seg_model_name, image_height, image_width, is_kitti=False, flow_model_name=None, pose_model_name=None):
        # import ipdb;ipdb.set_trace()
        # ! 论文中图2的matching network和Pose network
        self.vonet = VONet()    # matching network和pose network

        # load VO model separately (flow + pose) or at once
        # 加载训练好的matching network和pose network参数
        if flow_model_name.endswith('.pkl') and pose_model_name.endswith('.pkl'):
            modelname = 'models/' + flow_model_name
            self.load_vo_model(self.vonet.flowNet, modelname)
            modelname = 'models/' + pose_model_name
            self.load_vo_model(self.vonet.flowPoseNet, modelname)
        else:
            modelname = 'models/' + vo_model_name
            self.load_vo_model(self.vonet, modelname)

        self.vonet.cuda()   # 将网络的所有参数和权重存储在gpu上

        self.test_count = 0
        self.pose_norm = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

        # load the segmentation model
        self.testres = 1.2  # resolution，用于调整运动分割输入图像的大小
        if is_kitti:
            maxw, maxh = [int(self.testres * 1280), int(self.testres * 384)]
        else:
            maxw, maxh = [int(self.testres * 1024), int(self.testres * 448)]
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64
        maxh = max_h
        maxw = max_w

        # ! 论文中图2的motion segmentation network
        self.segnet = SegNet([1, maxw, maxh], md=[4, 4, 4, 4, 4], fac=1, exp_unc=not ('kitti' in seg_model_name))
        segmodelname = 'models/' + seg_model_name
        self.segnet = self.load_seg_model(self.segnet, segmodelname)
        
        self.segnet.cuda()

        self.segnet_initialize = False

        # To resize/crop segmentation mask
        self.resizedata = ResizeData(size=(image_height,1226)) if is_kitti else None
        self.cropdata = CropCenter((image_height, image_width))

        # To transform coordinates from NED to Blender
        Ry90 = np.array([[0,0,1,0], [0,1,0,0], [-1,0,0,0], [0,0,0,1]])
        Rx90 = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]])
        self.camT = Rx90.dot(Ry90)

        self.sigmoid = lambda x: 1/(1 + np.exp(-x))

    # 加载训练好的视觉里程计模型(参数权重)
    def load_vo_model(self, model, modelname):
        preTrainDict = torch.load(modelname)    # 加载训练好的模型
        model_dict = model.state_dict()         # 读取当前模型的权重参数字典
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict} # 保存当前模型所需的预训练权重参数

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)     # 将预训练模型的权重参数更新到当前模型的权重参数字典中
        model.load_state_dict(model_dict)       # 将更新后的权重参数字典里的参数加载到模型里去 
        print('VO Model %s loaded...' % modelname)
        return model

    def load_seg_model(self, model, modelname):
        model = nn.DataParallel(model, device_ids=[0])
        preTrainDict = torch.load(modelname, map_location='cpu')
        self.mean_L = preTrainDict['mean_L']
        self.mean_R = preTrainDict['mean_R']
        preTrainDict['state_dict'] = {k:v for k,v in preTrainDict['state_dict'].items()}
        model.load_state_dict(preTrainDict['state_dict'], strict=False)
        print('Segmentation Model %s loaded...' % modelname)
        return model

    # dytanvo网络的优化过程
    # sample：连续的两帧图片的字典，有5个key{img1,img2,intrinsic,img1_raw,img2_raw}
    # intrinstics：用于计算内参的focalx, centerx, centery, baseline
    # seg_thresh:motion segmentation的阈值
    def test_batch(self, sample, intrinsics, seg_thresh, iter_num):
        print("="*20)
        self.test_count += 1
        
        # import ipdb;ipdb.set_trace()
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        intrinsic = sample['intrinsic'].cuda()  # intrinsic layer

        # detach()创建一个pytorch的tensor
        # numpy()将tensor转为numpy
        # squeeze()去除不需要的维度
        img0_raw = sample['img1_raw'].detach().numpy().squeeze()
        img1_raw = sample['img2_raw'].detach().numpy().squeeze()

        # 只运行一次
        if not self.segnet_initialize:
            self.vonet.eval()       # 将模型model设为评估模式。对比：self.vonet.train()设置为训练模式
            self.segnet.eval()     
            self.initialize_segnet_input(img0_raw, intrinsics)
            self.segnet_initialize = True
        
        # with 是 Python 中的一个关键字，用于创建上下文管理器（context manager）
        # with 语句的基本用法是将需要执行的代码块包含在 with 语句中，并在其后面跟随一个上下文管理器。
        # torch.no_grad() 是一个上下文管理器，用于在特定代码块中禁用梯度计算。
        # 因为只有在训练阶段需要梯度计算，更新权重
        with torch.no_grad():
            imgL_noaug, imgLR = self.transform_segnet_input(img0_raw, img1_raw)
            flowdc = self.segnet.module.forward_VCN(imgLR)
            
            total_time = 0
            start_time = time.time()
            flow_output, _ = self.vonet([img0, img1], only_flow=True)
            flownet_time = time.time() - start_time
            total_time += flownet_time

            print("Flownet time: %.2f" % flownet_time)

            seg_thresholds = np.linspace(seg_thresh, 0.95, iter_num - 1)[::-1]
            for iter in range(iter_num):
                flow = flow_output.clone()
                if iter == 0:
                    cow_sigma_range = (20, 60)
                    log_sigma_range = (np.log(cow_sigma_range[0]), np.log(cow_sigma_range[1]))
                    cow_prop_range = (0.3, 0.6)
                    segmask = cow_masks(flow.shape[-2:], log_sigma_range, cow_sigma_range[1], cow_prop_range).astype(np.float32)
                    segmask = segmask[None,None,...]
                    segmask = torch.from_numpy(np.concatenate((segmask,) * img0.shape[0], axis=0)).cuda()

                start_time = time.time()
                _, pose_output = self.vonet([img0, img1, intrinsic, flow, segmask], only_pose=True)
                posenet_time = time.time() - start_time
                total_time += posenet_time

                print("Iter %d, Posenet time: %.2f; " % (iter, posenet_time), end='')

                # Do not pass segnet in the last iteration
                if iter == iter_num - 1:
                    break

                seg_thresh = seg_thresholds[iter] if iter < iter_num-1 else seg_thresh
                pose_input = pose_output.data.cpu().detach().numpy().squeeze()
                pose_input = pose_input * self.pose_norm
                pose_input = self.camT.T.dot(se2SE(pose_input)).dot(self.camT)
                
                start_time = time.time()
                disc_aux = [self.intr_list, imgL_noaug, pose_input[:3,:]]
                fgmask = self.segnet(imgLR, disc_aux, flowdc)
                segnet_time = time.time() - start_time
                total_time += segnet_time
                
                fgmask = cv2.resize(fgmask.cpu().numpy(), (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                fg_probs = self.sigmoid(fgmask)
                segmask = np.zeros(fgmask.shape[:2])
                segmask[fg_probs > seg_thresh] = 1.0

                # Resize/Crop segmask (Resize + Crop + Downscale 1/4)
                dummysample = {'segmask': segmask}
                if self.resizedata is not None:
                    dummysample = self.resizedata(dummysample)
                dummysample = self.cropdata(dummysample)
                segmask = dummysample['segmask']
                segmask = cv2.resize(segmask, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                segmask = segmask[None,None,...].astype(np.float32)
                segmask = torch.from_numpy(np.concatenate((segmask,) * img0.shape[0], axis=0)).cuda()

                print("Segnet time: %.2f" % segnet_time)
            
            posenp = pose_output.data.cpu().detach().numpy().squeeze()
            posenp = posenp * self.pose_norm  # The output is normalized during training, now scale it back
            flownp = flow.data.cpu().detach().numpy().squeeze()
            flownp = flownp * self.flow_norm

        # # calculate scale from GT posefile
        # if 'motion' in sample:
        #     motions_gt = sample['motion']
        #     scale = np.linalg.norm(motions_gt[:,:3], axis=1)
        #     trans_est = posenp[:,:3]
        #     trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
        #     posenp[:,:3] = trans_est 
        # else:
        #     print('    scale is not given, using 1 as the default scale value..')

        print("\n{} Pose inference using {}s: \n{}\n".format(self.test_count, total_time, posenp))

        return posenp, flownp

    def initialize_segnet_input(self, imgL_o, intrinsics):
        maxh = imgL_o.shape[0] * self.testres
        maxw = imgL_o.shape[1] * self.testres
        self.max_h = int(maxh // 64 * 64)       # 先计算maxh//64,计算图片大小可以划分为多少个64,比如300//64=4
        self.max_w = int(maxw // 64 * 64)       # 然后算*64,比如300//64*4=256
        if self.max_h < maxh: 
            self.max_h += 64
        if self.max_w < maxw: 
            self.max_w += 64
        self.input_size = imgL_o.shape

        # modify module according to inputs
        # 用nn.DataParallel来进行多GPU训练时，模型会被复制到每个GPU上进行并行计算。
        # 在这种情况下，net是一个带有DataParallel封装的模型。net.module是这个带有DataParallel封装的模型的原始模型
        for i in range(len(self.segnet.module.reg_modules)):
            self.segnet.module.reg_modules[i] = flow_reg([1, self.max_w//(2**(6-i)), self.max_h//(2**(6-i))], 
                            ent=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(self.segnet.module.warp_modules)):
            self.segnet.module.warp_modules[i] = WarpModule([1, self.max_w//(2**(6-i)), self.max_h//(2**(6-i))]).cuda()

        # foramt intrinsics input
        fl, cx, cy, bl = intrinsics
        fl_next = fl  # assuming focal length remains the same across frames
        self.intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        self.intr_list.append(torch.Tensor([self.input_size[1] / self.max_w]).cuda()) # delta fx
        self.intr_list.append(torch.Tensor([self.input_size[0] / self.max_h]).cuda()) # delta fy
        self.intr_list.append(torch.Tensor([fl_next]).cuda())

    def transform_segnet_input(self, imgL_o, imgR_o):
        imgL = cv2.resize(imgL_o, (self.max_w, self.max_h))
        imgR = cv2.resize(imgR_o, (self.max_w, self.max_h))
        imgL_noaug = torch.Tensor(imgL / 255.)[np.newaxis].float().cuda()
        
        # flip channel, subtract mean
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(self.mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(self.mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        imgLR = torch.cat([imgL,imgR],0)

        return imgL_noaug, imgLR