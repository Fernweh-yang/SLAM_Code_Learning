"""
    Brief: Demo for render labelled car 3d poses to the image
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import argparse
import cv2
import car_models
import numpy as np
import json
import pickle as pkl

import utils.data as data
import utils.utils as uts
import utils.eval_utils as eval_uts
import renderer.render_egl as render

# ! 加载Pickle文件需要，但后面操作有问题，所以改用json加载汽车模型
# from objloader import CHJ_tiny_obj

# ! OrderedDict 是python标准库提供的一个字典的子类
# 传统的字典键值对的顺序是随机的，OrderedDict按照键值对插入的顺序进行排列。
from collections import OrderedDict

# ! logging 模块将日志记录到 sys.stderr，即标准错误输出。这意味着日志信息将被打印到控制台。
import logging
logger = logging.getLogger()    # 创建一个名为 logger 的日志记录器
logger.setLevel(logging.INFO)   # 有不同级别的日志：DEBUG、INFO、WARNING、ERROR 和 CRITICAL。


class CarPoseVisualizer(object):
    def __init__(self, args=None, scale=0.4, linewidth=0.):
        """Initializer
        Input:
            scale: whether resize the image in case image is too large
            linewidth: 0 indicates a binary mask, while > 0 indicates
                       using a frame.
        """
        self.dataset = data.ApolloScape(args)                   # 得到数据集的内外参，图片高宽，立体校正映射
        self._data_config = self.dataset.get_3d_car_config()    # 得到数据集各个文件夹的地址

        self.MAX_DEPTH = 1e4
        self.MAX_INST_NUM = 100
        h, w = self._data_config['image_size']  # [2710, 3384]

        # must round prop to 4 due to renderer requirements
        # this will change the original size a bit, we usually need rescale
        # due to large image size
        # 需要改变图像大小时，对resize后的值进行一个向上求整的操作 [1084 1356]
        self.image_size = np.uint32(uts.round_prop_to(
            np.float32([h * scale, w * scale])))

        self.scale = scale          # 0.4 
        self.linewidth = linewidth  # 0
        self.colors = np.random.random((self.MAX_INST_NUM, 3)) * 255  # 100x3的数组，每个元素是[0,1]*255的值

    # ! 这个函数问题大大的，别用！！！用下面的load_car_models_json()来读取汽车模型
    def load_car_models(self):
        """Load all the car models from pickle
        """
        self.car_models = OrderedDict([])
        # 注意下面的car_models是另一个python脚本的名字，models是一个列表，保存了所有车的模型的id
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            # ../apolloscape/sample/car_models/aodi-a6.pkl
            car_model = '%s%s.pkl' % (self._data_config['car_model_dir'],  
                                       model.name)
            with open(car_model,"rb") as f:
                self.car_models[model.name] = pkl.load(f)
                # fix the inconsistency between obj and pkl
                self.car_models[model.name]['vertices'][:, [0, 1]] *= -1

    def load_car_models_json(self):
        """Load all the car models from json
        """
        self.car_models = OrderedDict([])
        # 注意下面的car_models是另一个python脚本的名字，models是一个列表，保存了所有车的模型的id
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            # ../apolloscape/sample/car_models/aodi-a6.pkl
            car_model = '%s%s.json' % (self._data_config['car_model_json_dir'],  
                                       model.name)
            with open(car_model) as json_file:
                self.car_models[model.name] = json.load(json_file)
                # fix the inconsistency between obj and pkl
                self.car_models[model.name]['vertices'] = np.array(self.car_models[model.name]['vertices'])     # 先将list转为np.array
                self.car_models[model.name]['vertices'][:,[0,1]] *= -1                                          # 然后再切片

    def render_car(self, pose, car_name):
        """Render a car instance given pose and car_name
        """
        car = self.car_models[car_name]     # 得到之前通过json读取到的汽车模型
        scale = np.ones((3, ))
        # ? 这个每个车的位姿是基于什么坐标系的呢？
        pose = np.array(pose)               # 照片中每个车的位姿，将列表表示的6维位姿转为np.array 
        vert = uts.project(pose, scale, car['vertices'])    # ! 将汽车模型各个顶点的坐标不知道转到什么坐标系下了？盲猜是相对于相机当前位姿
        K = self.intrinsic
        intrinsic = np.float64([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        depth, mask = render.renderMesh_py(np.float64(vert),
                                           np.float64(car['faces']),
                                           intrinsic,
                                           self.image_size[0],
                                           self.image_size[1],
                                           np.float64(self.linewidth))
        return depth, mask

    def compute_reproj_sim(self, car_names, out_file=None):
        """Compute the similarity matrix between every pair of cars.
        """
        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')

    def compute_reproj(self, car_name1, car_name2):
        """Compute reprojection error between two cars
        """
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0,5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)

    def merge_inst(self,
                   depth_in,
                   inst_id,
                   total_mask,
                   total_depth):
        """Merge the prediction of each car instance to a full image
        """

        render_depth = depth_in.copy()
        render_depth[render_depth <= 0] = np.inf
        depth_arr = np.concatenate([render_depth[None, :, :],
                                    total_depth[None, :, :]], axis=0)
        idx = np.argmin(depth_arr, axis=0)

        total_depth = np.amin(depth_arr, axis=0)
        total_mask[idx == 0] = inst_id

        return total_mask, total_depth


    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """
        #  返回一个修改图像大小后的内参矩阵
        intrinsic_out = uts.intrinsic_vec_to_mat(intrinsic, self.image_size)
        hs, ws = self.image_size    # [1084 1356]
        image_out = cv2.resize(image.copy(), (ws, hs))  # 将图像的大小调整为指定的宽度和高度

        return image_out, intrinsic_out


    def showAnn(self, image_name):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        car_pose_file = '%s%s.json' % (    
            self._data_config['pose_dir'], image_name)  # ../apolloscape/sample/car_poses/180116_053947113_Camera_5.json
        # car_poses是一个列表，列表元素是一个个字典，表示这张图里每个车的id和位姿
        # [{'car_id': 0, 'pose': [0.1584375649373483, 0.13563088205454682, -3.077314776688766, 
        #                        6.809990234375, 5.75218994140625, 25.004599609375]}
        #  .....]
        with open(car_pose_file) as f:
            car_poses = json.load(f)       # 得到每张照片中所有车的位姿:

        image_file = '%s%s.jpg' % (self._data_config['image_dir'], image_name) # ../apolloscape/sample/images/180116_053947113_Camera_5.jpg
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]    # (2710, 3384, 3) 读取图像并将其第三个通道颜色反转。 比如(255, 0, 0)红色，反转后的颜色值为 (0, 0, 255)蓝色

        # intrinsic are all used by Camera 5
        intrinsic = self.dataset.get_intrinsic(image_name, 'Camera_5')      # 得到4维内参向量fx,fy,c_x,c_y： array([0.68101296, 0.85087663, 0.49829724, 0.49999441])
        image, self.intrinsic = self.rescale(image, intrinsic)              # 得到修改大小后的图像(1084, 1356, 3)，和内参矩阵

        self.depth = self.MAX_DEPTH * np.ones(self.image_size)  # 给每个像素，初始化一个深度
        self.mask = np.zeros(self.depth.shape)                  # 给每个像素，初始化一个掩码

        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name  # 通过id转换到汽车的名字
            depth, mask = self.render_car(car_pose['pose'], car_name)   # 得到深度图和掩码
            self.mask, self.depth = self.merge_inst(
                depth, i + 1, self.mask, self.depth)

        self.depth[self.depth == self.MAX_DEPTH] = -1.0
        image = 0.5 * image
        for i in range(len(car_poses)):
            frame = np.float32(self.mask == i + 1)
            frame = np.tile(frame[:, :, None], (1, 1, 3))
            image = image + frame * 0.5 * self.colors[i, :]

        uts.plot_images({'image_vis': np.uint8(image),
                         'depth': self.depth, 'mask': self.mask},
                         layout=[1, 3])

        return image, self.mask, self.depth


class LabelResaver(object):
    """ Resave the raw labeled file to the required json format for evaluation
    """
    #(TODO Peng) Figure out why running pdb it is correct, but segment fault when
    # running
    def __init__(self, args):
        self.visualizer = CarPoseVisualizer(args, scale=0.5)
        self.visualizer.load_car_models()


    def strs_to_mat(self, strs):
        """convert str to numpy matrix
        """
        assert len(strs) == 4
        mat = np.zeros((4, 4))
        for i in range(4):
            mat[i, :] = np.array([np.float32(str_f) for str_f in strs[i].split(' ')])

        return mat


    def read_car_pose(self, file_name):
        """load the labelled car pose
        """
        cars = []
        lines = [line.strip() for line in open(file_name)]
        i = 0
        while i < len(lines):
            car = OrderedDict([])
            line = lines[i].strip()
            if 'Model Name :' in line:
                car_name = line[len('Model Name : '):]
                car['car_id'] = car_models.car_name2id[car_name].id
                pose = self.strs_to_mat(lines[i + 2: i + 6])
                pose[:3, 3] = pose[:3, 3] / 100.0 # convert cm to meter
                rot = uts.rotation_matrix_to_euler_angles(
                        pose[:3, :3], check=False)
                trans = pose[:3, 3].flatten()
                pose = np.hstack([rot, trans])
                car['pose'] = pose
                i += 6
                cars.append(car)
            else:
                i += 1

        return cars


    def convert(self, pose_file_in, pose_file_out):
        """ Convert the raw labelled file to required json format
        Input:
            file_name: str filename
        """
        car_poses = self.read_car_pose(pose_file_in)
        car_num = len(car_poses)
        MAX_DEPTH = self.visualizer.MAX_DEPTH
        image_size = self.visualizer.image_size
        intrinsic = self.visualizer.dataset.get_intrinsic(
                pose_file_in, 'Camera_5')
        self.visualizer.intrinsic = uts.intrinsic_vec_to_mat(intrinsic,
                image_size)
        self.depth = MAX_DEPTH * np.ones(image_size)
        self.mask = np.zeros(self.depth.shape)
        vis_rate = np.zeros(car_num)

        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            depth, mask = self.visualizer.render_car(car_pose['pose'], car_name)
            self.mask, self.depth = self.visualizer.merge_inst(
                depth, i + 1, self.mask, self.depth)
            vis_rate[i] = np.float32(np.sum(mask == (i + 1))) / (
                    np.float32(np.sum(mask)) + np.spacing(1))

        keep_idx = []
        for i, car_pose in enumerate(car_poses):
            area = np.round(np.float32(np.sum(
                self.mask == (i + 1))) / (self.visualizer.scale ** 2))
            if area > 1:
                keep_idx.append(i)

            car_pose['pose'] = car_pose['pose'].tolist()
            car_pose['area'] = int(area)
            car_pose['visible_rate'] = float(vis_rate[i])
            keep_idx.append(i)

        car_poses = [car_poses[idx] for idx in keep_idx]
        with open(pose_file_out, 'w') as f:
            json.dump(car_poses, f, sort_keys=True, indent=4,
                    ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render car instance and convert car labelled files.')
    parser.add_argument('--image_name', default='180116_053947113_Camera_5',
                        help='the dir of ground truth')
    parser.add_argument('--data_dir', default='../apolloscape/sample/',
                        help='the dir of ground truth')
    parser.add_argument('--split', default='sample_data', help='split for visualization')
    args = parser.parse_args()
    assert args.image_name

    if False:
        # (TODO) Debugging correct but segment fault when running. Useless for visualization
        print('Test converter')
        pose_file_in = './test_files/%s.poses' % args.image_name
        pose_file_out = './test_files/%s.json' % args.image_name
        label_resaver = LabelResaver(args)
        label_resaver.convert(pose_file_in, pose_file_out)

    # import os
    # print(os.getcwd())

    print('Test visualizer')
    visualizer = CarPoseVisualizer(args)
    visualizer.load_car_models_json()
    visualizer.showAnn(args.image_name)



