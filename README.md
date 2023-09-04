# SLAM_Code_Learning
为做NeRF-based SLAM的毕设所读过的开源代码，尽量做到行行有注释。

## 完整的SLAM方案

- [CO-SLAM代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/Co-SLAM)

  - 是一个neural RGB-D SLAM。

  - 在场景表达用到了NeRF的思想：通过每一帧的采样像素点的5D坐标$(x,y,z,\theta,\phi)$经由encoding-decoding网络训练得到颜色和深度
  - 位姿和encoding-decoding网络的参数都由pytorch.adam()根据4个损失函数来优化

- [ORB-SLAM2代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/ORB_SLAM2)

  - 已改写代码，现在适用于最新的opencv4.7和Pangolin 0.8

## 神经辐射场NeRF

- [nerf-pytorch代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/nerf-pytorch)

  nerf原始论文的pytorch实现版本

- [f2-nerf代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/f2-nerf)

  未读

## 神经网络中的李群运算

- [lietorch代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/lietorch)

  实现了各种李群李代数对于pytorch的tensor的封装，并实现了这些李群李代数数据结构之间的运算。

- [theseus代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/theseus)

  - facebook开源的一个借助pytorch来实现非线性优化的库
  - 不仅封装了各个李群李代数的数据结构，还实现了各种非线性优化算法如GN,LM

## 运动分割

- [DytanVO代码注释](https://github.com/Fernweh-yang/SLAM_Code_Learning/tree/main/DytanVO)

  一种基于学习的的视觉里程计VO方法，可以在动态环境中实现运动分隔
