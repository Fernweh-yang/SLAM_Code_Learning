# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.
# Credit: Xiangwei Wang https://github.com/TimingSpace

import numpy as np
from scipy.spatial.transform import Rotation as R

def line2mat(line_data):
    '''
    12 -> 4 x 4
    '''
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def mat2line(mat_data):
    '''
    4 x 4 -> 12
    '''
    line_data = np.zeros(12)
    line_data[:]=mat_data[:3,:].reshape((12))
    return line_data

def motion2pose(data):
    '''
    data: N x 12
    all_pose: (N+1) x 12
    '''
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def pose2motion(data, skip=0):
    '''
    data: N x 12
    all_motion (N-1-skip) x 12
    '''
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1,12))
    for i in range(0,data_size-1-skip):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1+skip,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result
    
def SO2so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat
### can get wrong result
def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i,:]
        SE = se2SE(se)
        all_SE = all_SE*SE
    all_se = SE2se(all_SE)
    mean_se = all_se/se_datas.shape[0]
    return mean_se

def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(se_datas.reshape(se_datas.shape[0],se_datas.shape[1],se_datas.shape[2]*se_datas.shape[3]),(0,2,1))
    se_result = np.zeros((se_datas.shape[0],se_datas.shape[2]))
    for i in range(0,se_datas.shape[0]):
        mean_se = se_mean(se_datas[i,:,:])
        se_result[i,:] = mean_se
    return se_result

def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def ses2poses_quat(data):
    '''
    ses: N x 6
    '''
    data_size = data.shape[0]
    all_pose_quat = np.zeros((data_size+1,7))
    all_pose_quat[0,:] = np.array([0., 0., 0., 0., 0., 0., 1.])
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        quat = SO2quat(pose[0:3,0:3])
        all_pose_quat[i+1,:3] = np.array([pose[0,3], pose[1,3], pose[2,3]])
        all_pose_quat[i+1,3:] = quat      
    return all_pose_quat

def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas,mean_std=[[1],[1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    return rr.as_quat()

def quat2SO(quat_data):
    return R.from_quat(quat_data).as_matrix()


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


def pos_quats2SE_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3,0:3] = SO
        SE[0:3,3]   = quat[0:3]
        SEs.append(SE)
    return SEs

def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def SEs2ses(data):
    '''
    data: N x 12
    ses: N x 6
    '''
    data_size = data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        ses[i,:] = SE2se(line2mat(data[i]))
    return ses

def ses2SEs(data):
    '''
    data: N x 6
    SEs: N x 12
    '''
    data_size = data.shape[0]
    SEs = np.zeros((data_size,12))
    for i in range(0,data_size):
        SEs[i,:] = mat2line(se2SE(data[i]))
    return SEs

def SE2quat(SE_data):
    '''
    SE_data: 4 x 4
    quat: 7
    '''
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def quat2SE(quat_data):
    '''
    quat_data: 7
    SE: 4 x 4
    '''
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    return SE

def SEs2quats(SEs_data):
    '''
    SE_data: N x 12
    quat: N x 7
    '''
    data_len = SEs_data.shape[0]
    all_quats = np.zeros((data_len,7))
    for i in range(0,data_len):
        SE = line2mat(SEs_data[i])
        all_quats[i] = SE2quat(SE)
    return all_quats

def quats2SEs(quat_datas):
    '''
    pos_quats: N x 7
    SEs: N x 12
    '''
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = mat2line(SE)
    return SEs

def motion_ses2pose_quats(data):
    '''
    data: N x 6 motion data
    poses_quat: (N+1) x 7 pose data
    '''
    motions_SEs = ses2SEs(data) # N x 6 -> N x 12
    poses_SEs  = motion2pose(motions_SEs) # N x 12 -> (N + 1) x 12
    poses_quat = SEs2quats(poses_SEs) # (N + 1) x 12 -> (N+1) x 7
    return poses_quat

def pose_quats2motion_ses(data):
    '''
    data: N x 7 pose list
    motions: (N-1-skip) x 6 se3 list
    '''
    poses_SEs = quats2SEs(data) # N x 7 -> N x 12
    matrix = pose2motion(poses_SEs) # N x 12 -> (N-1-skip) x 12
    motions = SEs2ses(matrix).astype(np.float32) # (N-1-skip) x 12 -> (N-1-skip) x 6
    return motions

def kitti2tartan(traj):
    '''
    traj: in kitti style, N x 12 numpy array, in camera frame
    output: in TartanAir style, N x 7 numpy array, in NED frame
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pose.reshape(3,4)
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
        
    return np.array(new_traj)

def tartan2kitti(traj):
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pos_quat2SE(pose).reshape(3,4)
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(ttt[:3,:].reshape(12))
        
    return np.array(new_traj)