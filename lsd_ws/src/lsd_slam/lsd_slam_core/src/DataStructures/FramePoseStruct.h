/**
 * This file is part of LSD-SLAM.
 *
 * Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
 * For more information see <http://vision.in.tum.de/lsdslam>
 *
 * LSD-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LSD-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "util/SophusUtil.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"

namespace lsd_slam
{
    class Frame;
    class FramePoseStruct
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FramePoseStruct(Frame *frame);
        virtual ~FramePoseStruct();

        // parent, the frame originally tracked on. never changes.
        // 当前帧跟踪的参考帧的位姿
        FramePoseStruct *trackingParent;

        // set initially as tracking result (then it's a SE(3)),
        // and is changed only once, when the frame becomes a KF (->rescale).
        // 最初跟踪得到的位姿变换T，只有在该帧成为关键帧的时候会被修改
        // 在SE3Tracker.cpp中，这个值=当前帧的SE3位姿转为Sim3的值，其中仿射尺度=1
        Sim3 thisToParent_raw;

        int frameID;
        Frame *frame;

        // whether this poseStruct is registered in the Graph. if true MEMORY WILL BE HANDLED BY GRAPH
        // 如果为true:表示某一位姿已加入图
        bool isRegisteredToGraph;

        // whether pose is optimized (true only for KF, after first applyPoseGraphOptResult())
        // 只在对关键帧kf调用了applyPoseGraphOptResult()函数后为true, 表示该关键帧已经被图优化过了
        bool isOptimized;

        // true as soon as the vertex is added to the g2o graph.
        // 如果该帧的位姿被加入到了g2o的图优化中，就为true
        bool isInGraph;

        // graphVertex (if the frame has one, i.e. is a KF and has been added to the graph, otherwise 0).
        // 图优化的头vertex: 表示关键帧的位姿
        VertexSim3 *graphVertex;

        void setPoseGraphOptResult(Sim3 camToWorld);    // 设置位姿图优化的结果
        void applyPoseGraphOptResult();                 // 将位姿图优化的结果应用到当前帧上
        Sim3 getCamToWorld(int recursionDepth = 0);     // 获取相机到世界坐标系的变换
        void invalidateCache();                         // 使缓存失效?

    private:
        // 缓存计数器？
        int cacheValidFor;
        static int cacheValidCounter;

        // absolute position (camToWorld).
        // can change when optimization offset is merged.
        // 相机到世界坐标系的位姿变换
        Sim3 camToWorld;

        // new, optimized absolute position. is added on mergeOptimization.
        // 图优化后的相机到世界坐标系的位姿变换
        Sim3 camToWorld_new;

        // whether camToWorld_new is newer than camToWorld
        // 表示 camToWorld_new 是否比 camToWorld 更新。
        bool hasUnmergedPose;
    };

} // namespace lsd_slam
