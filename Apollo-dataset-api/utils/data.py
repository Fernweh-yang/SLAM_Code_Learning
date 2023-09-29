"""
    Brief: Apollo dataset path and parameters config class
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import cv2
import numpy as np
import utils as uts


class ApolloScape(object):
    def __init__(self, args=None, scale=1.0, use_stereo=False):
        self._data_dir = './apolloscape/'
        """
        args为:
        Setting(image_name='180116_053947113_Camera_5', data_dir='/home/yang/Datasets/ApolloScape/CarInstance/sample/')
        数据结构是一个pytorch提供的collections.namedtuple封装的元组
        """
        self._args = args
        self._scale = scale         # 若不为1，则是要resize图片
        self._get_data_parameters() # 得到数据集的图片高宽，两相机的内参外参
        if use_stereo:
            self._get_stereo_rectify_params()

    def _get_data_parameters(self):
        """get the data configuration of the dataset.
           These parameters are shared across different tasks
        """
        self._data_config = {}
        self._data_config['image_size_raw'] = [2710, 3384]  # 数据集的每张图都是height:2710,width:3384
        # when need to rescale image due to large data
        self._data_config['image_size'] = [int(2710 * self._scale),
                                           int(3384 * self._scale)]

        # 相机内参fx, fy, cx, cy
        self._data_config['intrinsic'] = {
            'Camera_5': np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array(
                [2300.39065314361, 2301.31478860597,
                 1713.21615190657, 1342.91100799715])}

        # normalized intrinsic for handling image resizing
        # 归一化内参
        cam_names = self._data_config['intrinsic'].keys()
        for c_name in cam_names:
            self._data_config['intrinsic'][c_name][[0, 2]] /= \
                self._data_config['image_size_raw'][1]  # [1]: width = 3384
            self._data_config['intrinsic'][c_name][[1, 3]] /= \
                self._data_config['image_size_raw'][0]  # [0]: height = 2710

        # relative pose of camera 6 wrt camera 5
        self._data_config['extrinsic'] = {
            'R': np.array([
                [9.96978057e-01, 3.91718762e-02, -6.70849865e-02],
                [-3.93257593e-02, 9.99225970e-01, -9.74686202e-04],
                [6.69948100e-02, 3.60985263e-03, 9.97746748e-01]]),
            'T': np.array([-0.6213358, 0.02198739, -0.01986043])
        }

        """
        crop margin after stereo rectify for getting the region
        with stereo matching, however it can remove some valid region
        
        stereo rectify 立体校正: 把实际中非共面行对准的两幅图像，校正成共面行对准，提高匹配搜索的效率,因为二维搜索变为一维搜索

        """
        self._data_config['stereo_crop'] = np.array(
            [1232., 668., 2500., 2716.])  # 分别为图像顶部、左侧、底部、右侧的裁减边距
        self._data_config['stereo_crop'][[0, 2]] /= \
            self._data_config['image_size_raw'][0]  # [0]: height = 2710
        self._data_config['stereo_crop'][[1, 3]] /= \
            self._data_config['image_size_raw'][1]  # [1]: width = 3384

    def _get_stereo_rectify_params(self):
        """ if using stereo, we need to findout the stereo parameters,
            based on the extrinsic parameters for the two cameras
        """
        # 把内参fx, fy, cx, cy写成矩阵的形式：K
        camera_names = self._data_config['intrinsic'].keys()
        camera5_mat = uts.intrinsic_vec_to_mat(
            self._data_config['intrinsic']['Camera_5'],
            self._data_config['image_size'])
        camera6_mat = uts.intrinsic_vec_to_mat(
            self._data_config['intrinsic']['Camera_6'],
            self._data_config['image_size'])

        distCoeff = np.zeros(4)
        image_size = (self._data_config['image_size'][1],   # [1]: width = 3384*resize
                      self._data_config['image_size'][0])   # [0]: height = 2710*resize

        """
        进行立体校正:
        R1：校正后的左相机旋转矩阵
        R2：校正后的右相机旋转矩阵
        P1：校正后的左相机投影矩阵
        P2：校正后的右相机投影矩阵
        Q：立体校正矩阵
        validPixROI1：校正后左相机有效像素区域
        validPixROI2：校正后右相机有效像素区域
        """
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=camera5_mat,              # 左相机内参矩阵
            distCoeffs1=distCoeff,                  # 左相机畸变系数
            cameraMatrix2=camera6_mat,              # 右相机内参矩阵
            distCoeffs2=distCoeff,                  # 右相机畸变系数
            imageSize=image_size,                   # 图像的大小
            R=self._data_config['extrinsic']['R'],  # 两个相机的旋转矩阵
            T=self._data_config['extrinsic']['T'],  # 两个相机之间的平移向量
            flags=cv2.CALIB_ZERO_DISPARITY,         # 校正方法
            alpha=1)

        
        """
        for warping image 5
        cv2.initUndistortRectifyMap 函数用于生成映射表，用于将图像中的像素坐标转换到校正后的坐标系中。
        校正后的坐标系是没有畸变的，因此可以将图像中的像素坐标转换到校正后的坐标系中，然后进行后续处理，例如三维重建、图像测量等。

        mapx：x方向的映射表
        mapy：y方向的映射表
        """
        map1x, map1y = cv2.initUndistortRectifyMap(
            cameraMatrix=camera5_mat,   # 相机内参矩阵
            distCoeffs=distCoeff,       # 相机畸变系数
            R=R1,                       # 校正后的相机旋转矩阵
            newCameraMatrix=P1,         # 校正后的相机投影矩阵
            size=image_size,            # 图像的大小
            m1type=cv2.CV_32FC1)

        # for warping image 6
        map2x, map2y = cv2.initUndistortRectifyMap(
            cameraMatrix=camera6_mat,
            distCoeffs=distCoeff,
            R=R2,
            newCameraMatrix=P2,
            size=image_size,
            m1type=cv2.CV_32FC1)

        res = {'Camera_5_rot': R1,      # 左相机旋转矩阵
               'Camera_5_intr': P1,     # 左相机投影矩阵KT
               'Camera_5_mapx': map1x,  # 左相机x方向的映射表
               'Camera_5_mapy': map1y,  # 左相机y方向的映射表
               'Camera_6_rot': R2,
               'Camera_6_intr': P2,
               'Camera_6_mapx': map2x,
               'Camera_6_mapy': map2y}

        # get new intrinsic and rotation parameters after rectification
        for name in camera_names:
            res[name + '_intr'] = uts.intrinsic_mat_to_vec(res[name + '_intr'])
            res[name + '_intr'][[0, 2]] /= self._data_config['image_size'][1]
            res[name + '_intr'][[1, 3]] /= self._data_config['image_size'][0]
            rect_extr_mat = np.eye(4)
            rect_extr_mat[:3, :3] = res[name + '_rot']
            res[name + '_ext'] = rect_extr_mat

        self._data_config.update(res)

    def stereo_rectify(self, image, camera_name,
                       interpolation=cv2.INTER_LINEAR):
        """ Given an image we rectify this image for stereo matching
        Input:
            image: the input image
            camera_name: 'Camera_5' or 'Camera_6' the name of camera where the
                    image is
            interpolation: the method of warping
        Output:
            the rectified image
        """
        image_rect = cv2.remap(image,
                               self._data_config[camera_name + '_mapx'],
                               self._data_config[camera_name + '_mapy'],
                               interpolation)
        return image_rect

    def get_3d_car_config(self):
        """get configuration of the dataset for 3d car understanding
        """
        ROOT = self._data_dir + '3d_car_instance/' if self._args is None else \
            self._args.data_dir
        split = self._args.split if hasattr(
            self._args, 'split') else 'sample_data'

        self._data_config['image_dir'] = ROOT + '%s/images/' % split
        self._data_config['pose_dir'] = ROOT + '%s/car_poses/' % split
        self._data_config['train_list'] = ROOT + '%s/split/train.txt'
        self._data_config['val_list'] = ROOT + '%s/split/val.txt'

        self._data_config['car_model_dir'] = ROOT + 'car_models/'

        return self._data_config

    def get_self_local_config(self, Road, split):
        """get configuration of the dataset for 3d car understanding
        """
        ROOT = self._data_dir + 'self_localization/' if self._args is None else \
            self._args.data_dir
        split = self._args.split if hasattr(self._args, 'split') else 'sample_data'

        self._data_config['image_dir'] = ROOT + '%s/%s/image/' % (split, Road)
        self._data_config['pose_dir'] = ROOT + '%s/%s/pose/' % (split, Road)
        self._data_config['cloud_dir'] = ROOT + '%s/%s/point_cloud/' % (split, Road)

        return self._data_config


    def get_intrinsic(self, image_name, camera_name=None):
        assert self._data_config
        if camera_name:
            return self._data_config['intrinsic'][camera_name]
        else:
            for name in self._data_config['intrinsic'].keys():
                if name in image_name:
                    return self._data_config['intrinsic'][name]
        raise ValueError('%s has no provided intrinsic' % image_name)


