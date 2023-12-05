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

#include <DataStructures/FramePoseStruct.h>
#include "DataStructures/Frame.h"

namespace lsd_slam
{
    int FramePoseStruct::cacheValidCounter = 0;

    int privateFramePoseStructAllocCount = 0;

    FramePoseStruct::FramePoseStruct(Frame *frame)
    {
        cacheValidFor = -1;
        isOptimized = false;
        thisToParent_raw = camToWorld = camToWorld_new = Sim3();
        this->frame = frame;
        frameID = frame->id();
        trackingParent = 0;
        isRegisteredToGraph = false;
        hasUnmergedPose = false;
        isInGraph = false;

        this->graphVertex = nullptr;

        privateFramePoseStructAllocCount++;
        if (enablePrintDebugInfo && printMemoryDebugInfo)
            printf("ALLOCATED pose %d, now there are %d\n", frameID, privateFramePoseStructAllocCount);
    }

    FramePoseStruct::~FramePoseStruct()
    {
        privateFramePoseStructAllocCount--;
        if (enablePrintDebugInfo && printMemoryDebugInfo)
            printf("DELETED pose %d, now there are %d\n", frameID, privateFramePoseStructAllocCount);
    }

    void FramePoseStruct::setPoseGraphOptResult(Sim3 camToWorld)
    {
        if (!isInGraph)
            return;

        camToWorld_new = camToWorld;
        hasUnmergedPose = true;
    }

    // ! 更新g2o优化完后的结果
    void FramePoseStruct::applyPoseGraphOptResult()
    {
        if (!hasUnmergedPose)
            return;

        camToWorld = camToWorld_new;
        isOptimized = true;
        hasUnmergedPose = false;
        cacheValidCounter++;
    }
    void FramePoseStruct::invalidateCache()
    {
        cacheValidFor = -1;
    }

    // ! 不停的递归相对位姿，直到得到当前帧相对于世界坐标系的位姿，即绝对位姿
    // 在.h文件中recursionDepth默认为0，所以在跟踪线程中都是直接getCamToWorld()没加参数来调用
    Sim3 FramePoseStruct::getCamToWorld(int recursionDepth)
    {
        // prevent stack overflow 确保递归深度不会超过5000
        assert(recursionDepth < 5000);

        // 如果当前帧已经被加入图优化了，它的绝对位姿只能由BA优化修改
        // if the node is in the graph, it's absolute pose is only changed by optimization.
        if (isOptimized)
            return camToWorld;

        // 如果缓存的位姿依然有效，就直接返回，避免重复计算
        // return chached pose, if still valid.
        if (cacheValidFor == cacheValidCounter)
            return camToWorld;

        // 如果没有父节点，说明是第一帧，直接返回0矩阵Sim3()
        // return id if there is no parent (very first frame)
        if (trackingParent == nullptr)
            return camToWorld = Sim3();

        /*
            如果以上条件都不满足，说明当前帧有父帧，缓存的位姿过时了且未经优化。
            此时执行递归:
                getCamToWorld(recursionDepth + 1)得到父帧相对于世界坐标系的绝对位姿
                乘上 thisToParent_raw当前帧相对于父帧的相对位姿
        */
        // abs. pose is computed from the parent's abs. pose, and cached.
        cacheValidFor = cacheValidCounter;
        return camToWorld = trackingParent->getCamToWorld(recursionDepth + 1) * thisToParent_raw;
    }

} // namespace lsd_slam
