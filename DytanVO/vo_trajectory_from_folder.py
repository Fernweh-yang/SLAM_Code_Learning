from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, ResizeData, dataset_intrinsics, DownscaleFlow
from Datasets.utils import plot_traj, visflow, load_kiiti_intrinsics, load_sceneflow_extrinsics
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from evaluator.transformation import pose_quats2motion_ses, motion_ses2pose_quats
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.evaluator_base import per_frame_scale_alignment
from DytanVO import DytanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

def get_args():
    parser = argparse.ArgumentParser(description='Inference code of DytanVO')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained VO model (default: "")')
    parser.add_argument('--flow-model-name', default='',
                        help='name of pretrained flow model (default: "")')
    parser.add_argument('--pose-model-name', default='',
                        help='name of pretrained pose model (default: "")')
    parser.add_argument('--seg-model-name', default='',
                        help='name of pretrained segmentation model (default: "")')
    parser.add_argument('--airdos', action='store_true', default=False,
                        help='airdos test (default: False)')
    parser.add_argument('--rs_d435', action='store_true', default=False,
                        help='realsense d435i test (default: False)')
    parser.add_argument('--sceneflow', action='store_true', default=False,
                        help='sceneflow test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--commaai', action='store_true', default=False,
                        help='commaai test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--seg-thresh', type=float, default=0.7,
                        help='threshold for motion segmentation')
    parser.add_argument('--iter-num', type=int, default=2,
                        help='number of iterations')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    
    # *********************** 创建网络:start ***********************
    # 创建论文图2所示的网络结构
    testvo = DytanVO(args.vo_model_name, args.seg_model_name, args.image_height, args.image_width, 
                    args.kitti, args.flow_model_name, args.pose_model_name)
    # *********************** 创建网络:end ***********************


    # *********************** 加载数据集:start ***********************
    # 加载数据集
    if args.kitti:
        datastr = 'kitti'
    elif args.airdos:
        datastr = 'airdos'
    elif args.rs_d435:
        datastr = 'rs_d435'
    elif args.sceneflow:
        datastr = 'sceneflow'
    elif args.commaai:
        datastr = 'commaai'
    else:
        datastr = 'tartanair'
    # 设置不同数据集相机的内参K
    focalx, focaly, centerx, centery, baseline = dataset_intrinsics(datastr, '15mm' in args.test_dir) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
        focalx, focaly, centerx, centery, baseline = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    if datastr == 'kitti':
        # Compose(List[Transform])创建一个实例，它先将一个列表的图像变换整合到一起，并之后施加给某Image
        # ResizeData()将图片的大小调整为(args.image_height, 1226)
        # CropCenter()将图片裁减到指定的大小(args.image_height, args.image_width)，并确保图像位于原始图像的中心
        # ToTensor()将图片数据转换为pytorch的张量格式
        transform = Compose([ResizeData((args.image_height, 1226)), CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    else:
        # DownscaleFlow()将输入的字典中的 "flow"、"intrinsic" 和 "fmask" 数据进行降采样（Downscale）操作：即将数据的尺寸缩小为原来的一定比例。
        transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    # 加载测试数据集，得到一个存储了所有图片的实例
    # 返回得到一个dict={img1,img2,intrinsicLayer}的字典
    testDataset = TrajFolderDataset(args.test_dir, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    # DataLoader 是 PyTorch 中 torch.utils.data 模块中的一个类，它提供了数据加载和批处理的功能，用于加载训练集和测试集等数据，并将其组织成批量供模型训练或测试使用。
    '''
    使用dataLoader的步骤:
    1. 创建自定义Dataset,承自torch.utils.data.Dataset, 并实现__len__()和__getitem__()方法来获取数据样本和标签。即上面的TrajFolderDataset()
    2. 创建Dataset实例。即上面的testDataset
    3. 创建DataLoader实例。即下面的testDataloader
    4. 迭代数据。即下面的testDataiter
    在使用DataLoader加载数据时,每次加载都会调用Dataset即这里的TrajFolderDataset()的__getitem__()方法来获取一个批次(batch)的数据样本。
    '''
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    # 将 testDataloader 转换为迭代器
    testDataiter = iter(testDataloader)
# *********************** 加载数据集:end ***********************


# *********************** Maching net计算光流和运动:start ***********************
    motionlist = []
    # 生成一个测试名称（testname）的字符串，根据不同的输入数据和命令行参数组合而生成一个唯一标识的字符串。
    testname = datastr + '_' + args.vo_model_name.split('.')[0] + '_' + args.test_dir.split('/')[-1]
    # 如果要保存光流，设置保存地址
    if args.save_flow:
        flowdir = 'results/'+testname+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0
    while True:
        try:
            # ! 对应论文fig2的开头，提取2张照片。也计算了Intrinsics Layer
            # testDataiter转换为迭代器后，就可以方便的用next()函数获取下一个数据
            # 根据TrajFolderDataset()中定义的__getitem__()函数，sample是1个储存2张照片和内参层的字典{'img1': img1, 'img2': img2，'intrinsic'：intrinsicLayer }
            sample = testDataiter.next()
        except StopIteration:
            break
        # ! 对应论文fig2的matching network处
        # ! test_batch()中相继调用了论文中segmentation network,pose network等，并完成优化
        # 参见论文：送入matching network，得到运动和光流
        motion, flow = testvo.test_batch(sample, [focalx, centerx, centery, baseline], args.seg_thresh, args.iter_num)
        # 将每个像素的运动放入列表中
        motionlist.append(motion)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                # 将光流numpy数组flowk保存为二进制文件.npy,第一个变量是保存的名字
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                # 将光流先表示为hsv，再表示为rgb
                flow_vis = visflow(flowk)
                # 将光流可视化后的图像保存为xx.png
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1
                
    motions = np.array(motionlist)
# *********************** Maching net计算光流和运动:end ***********************


# *********************** 如果有真值就进行评估:start ***********************
    # calculate ATE, RPE, KITTI-RPE
    # 读取场景流(u,v,w)位姿的真值
    # 景流是三维场景中像素点在时间上的三维位移信息，用于描述像素在三维空间中的运动和深度变化。场景流是光流的三维推广
    if args.pose_file.endswith('.txt'):
        if datastr == 'sceneflow':
            # 如果是场景流，因为有左右相机，所以读取时也是左右分开处理的
            # pose由四元数+3d坐标表示
            gtposes = load_sceneflow_extrinsics(args.pose_file)
        else:
            # 如果不是场景流，就是简单的光流位姿
            gtposes = np.loadtxt(args.pose_file)
            if datastr == 'airdos':
                gtposes = gtposes[:,1:]  # remove the first column of timestamps
        
        # 将运动从四元数和三维坐标转换为SE(3)，即T(4x4)
        gtmotions = pose_quats2motion_ses(gtposes)
        # 使用真值修正预测的运动
        estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
        # 再重新将运动从T转为四元数
        estposes = motion_ses2pose_quats(estmotion_scale)

        # 评估网络效果
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(gtposes, estposes, scale=True, kittitype=(datastr=='kitti'))
        
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
# *********************** 如果有真值就进行评估:end ***********************


# *********************** 保存和可视化结果:start ***********************
        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt', motion_ses2pose_quats(motions))
# *********************** 保存和可视化结果:end ***********************
