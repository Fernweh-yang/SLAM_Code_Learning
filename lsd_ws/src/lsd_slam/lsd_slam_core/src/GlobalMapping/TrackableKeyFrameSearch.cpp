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

#include "TrackableKeyFrameSearch.h"

#include "GlobalMapping/KeyFrameGraph.h"
#include "DataStructures/Frame.h"
#include "Tracking/SE3Tracker.h"

namespace lsd_slam
{
    TrackableKeyFrameSearch::TrackableKeyFrameSearch(KeyFrameGraph *graph, int w, int h, Eigen::Matrix3f K) : graph(graph)
    {
        tracker = new SE3Tracker(w, h, K);

        fowX = 2 * atanf((float)((w / K(0, 0)) / 2.0f));
        fowY = 2 * atanf((float)((h / K(1, 1)) / 2.0f));

        msTrackPermaRef = 0;
        nTrackPermaRef = 0;
        nAvgTrackPermaRef = 0;

        if (enablePrintDebugInfo && printRelocalizationInfo)
            printf("Relocalization Values: fowX %f, fowY %f\n", fowX, fowY);
    }

    TrackableKeyFrameSearch::~TrackableKeyFrameSearch()
    {
        delete tracker;
    }

    // ! 在已经加入到图优化g2o中的关键帧中，寻找新关键帧frame的所有跟踪到的作为它的参考帧
    std::vector<TrackableKFStruct, Eigen::aligned_allocator<TrackableKFStruct>>
    TrackableKeyFrameSearch::findEuclideanOverlapFrames(Frame *frame, float distanceTH, float angleTH, bool checkBothScales)
    {
        // basically the maximal angle-difference in viewing direction is angleTH*(average FoV).
        // e.g. if the FoV is 130°, then it is angleTH*130°.
        // FOV 视场角（Field of View）：从摄像机或观察者位置朝视野中心看去的视线辐射出的角度
        // fowX 和 fowY 分别表示视场角在水平（X轴）和垂直（Y轴）方向的分量
        float cosAngleTH = cosf(angleTH * 0.5f * (fowX + fowY));

        Eigen::Vector3d pos = frame->getScaledCamToWorld().translation();                               // 得到变换矩阵的位移t
        Eigen::Vector3d viewingDir = frame->getScaledCamToWorld().rotationMatrix().rightCols<1>();      // 得到变换矩阵的旋转R

        std::vector<TrackableKFStruct, Eigen::aligned_allocator<TrackableKFStruct>> potentialReferenceFrames;   // 用于记录潜在的参考帧

        // 计算尺度
        float distFacReciprocal = 1;
        if (checkBothScales)
            distFacReciprocal = frame->meanIdepth / frame->getScaledCamToWorld().scale();

        // for each frame, calculate the rough score, consisting of pose, scale and angle overlap.
        // 启用共享锁lock_shared
        graph->keyframesAllMutex.lock_shared();
        // 遍历所有的关键帧
        for (unsigned int i = 0; i < graph->keyframesAll.size(); i++)
        {   
            // ***************** 计算两帧之间的距离 *****************
            // 得到要比较的帧的位移t
            Eigen::Vector3d otherPos = graph->keyframesAll[i]->getScaledCamToWorld().translation();
            // 计算当前关键帧和要比较的帧之间的距离
            // get distance between the frames, scaled to fit the potential reference frame.
            float distFac = graph->keyframesAll[i]->meanIdepth / graph->keyframesAll[i]->getScaledCamToWorld().scale();
            // 选取两个尺度因子中较小的那个
            if (checkBothScales && distFacReciprocal < distFac)
                distFac = distFacReciprocal;
            // 最终的距离 = 距离*尺度因子
            Eigen::Vector3d dist = (pos - otherPos) * distFac;
            // 将距离平方
            float dNorm2 = dist.dot(dist);
            // 如果距离大于了距离阈值，就跳出当前循环
            if (dNorm2 > distanceTH)
                continue;

            // ***************** 计算两帧之间的角度 *****************
            // 得到要比较帧的旋转
            Eigen::Vector3d otherViewingDir = graph->keyframesAll[i]->getScaledCamToWorld().rotationMatrix().rightCols<1>();
            // 计算当前关键帧和要比较的帧之间的旋转角度
            float dirDotProd = otherViewingDir.dot(viewingDir);
            // 如果角度小于角度阈值，就跳出当前循环
            if (dirDotProd < cosAngleTH)
                continue;

            // ***************** 如果距离和角度都符合要求，就记录在潜在的参考帧potentialReferenceFrames中 *****************
            potentialReferenceFrames.push_back(TrackableKFStruct());
            potentialReferenceFrames.back().ref = graph->keyframesAll[i];
            potentialReferenceFrames.back().refToFrame =
                se3FromSim3(graph->keyframesAll[i]->getScaledCamToWorld().inverse() * frame->getScaledCamToWorld()).inverse();
            potentialReferenceFrames.back().dist = dNorm2;
            potentialReferenceFrames.back().angle = dirDotProd;
        }
        // 关闭共享锁unlock_shared
        graph->keyframesAllMutex.unlock_shared();

        return potentialReferenceFrames;
    }

    // ! 在已经加入到图优化g2o中的关键帧中，寻找新关键帧frame的所有能跟踪到的作为它的参考帧，并计算该最新关键帧的分数(公式16)
    // ! 如果分数不够高(距离不够远)就要在这些能追踪到的参考帧里选一个作为当前关键帧的替代，因为论文中要求只有足够远才能成为关键帧
    Frame *TrackableKeyFrameSearch::findRePositionCandidate(Frame *frame, float maxScore)
    {   
        // ***************** 在已经加入到图优化g2o中的关键帧中，寻找新关键帧frame的所有跟踪到的作为它的参考帧 *****************
        std::vector<TrackableKFStruct, Eigen::aligned_allocator<TrackableKFStruct>> potentialReferenceFrames =
            findEuclideanOverlapFrames(frame, maxScore / (KFDistWeight * KFDistWeight), 0.75);

        float bestScore = maxScore;
        float bestDist, bestUsage;
        float bestPoseDiscrepancy = 0;
        Frame *bestFrame = 0;
        SE3 bestRefToFrame = SE3();
        SE3 bestRefToFrame_tracked = SE3();

        int checkedSecondary = 0;
        // 遍历所有潜在的参考帧
        for (unsigned int i = 0; i < potentialReferenceFrames.size(); i++)
        {   
            // 如果该参考帧是当前关键帧最先跟踪的帧(FramePoseStruct *trackingParent;)，那么跳出循环
            if (frame->getTrackingParent() == potentialReferenceFrames[i].ref)
                continue;
            // 前INITIALIZATION_PHASE_COUNT=5帧用于初始化，如果该参考帧的索引idxInKeyframes<5，那么跳出循环
            if (potentialReferenceFrames[i].ref->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
                continue;

            // ***************** 检查当前关键帧帧与参考帧之间的永久参考点(PermRef)的重叠度 *****************
            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);  // 系统调用，获取当前时间
            tracker->checkPermaRefOverlap(potentialReferenceFrames[i].ref, potentialReferenceFrames[i].refToFrame);
            gettimeofday(&tv_end, NULL);
            msTrackPermaRef = 0.9 * msTrackPermaRef + 0.1 * ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                             (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
            nTrackPermaRef++;
            // 根据参考帧和当前关键帧之间的距离dist和永久参考点的使用情况pointUsage，计算参考帧的分数
            float score = getRefFrameScore(potentialReferenceFrames[i].dist, tracker->pointUsage);
            // * 如果分数不够高(距离不够远)就要重新算分，如果分数比之前最好的要好，就把这个参考帧保存下来作为最新关键帧frame最好的参考帧(替代帧)
            if (score < maxScore)
            {

                // SE3Tracker::trackFrameOnPermaref()区别于SE3Tracker::trackFrame的精确计算，是一个快速计算位姿的方法
                SE3 RefToFrame_tracked =
                    tracker->trackFrameOnPermaref(potentialReferenceFrames[i].ref, frame, potentialReferenceFrames[i].refToFrame);
                // 重新计算距离dist
                Sophus::Vector3d dist = RefToFrame_tracked.translation() * potentialReferenceFrames[i].ref->meanIdepth;
                // 重新计算参考帧的分数
                float newScore = getRefFrameScore(dist.dot(dist), tracker->pointUsage);
                // 计算之前估计的位姿，和新计算的位姿RefToFrame_tracked之间的差异
                float poseDiscrepancy = (potentialReferenceFrames[i].refToFrame * RefToFrame_tracked.inverse()).log().norm();
                // 跟踪好的点占总数的百分比
                float goodVal = tracker->pointUsage * tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
                checkedSecondary++;

                // 判断新的位姿计算是否比之前最好的更好
                if (tracker->trackingWasGood && goodVal > relocalizationTH && newScore < bestScore && poseDiscrepancy < 0.2)
                {
                    bestPoseDiscrepancy = poseDiscrepancy;
                    bestScore = score;
                    bestFrame = potentialReferenceFrames[i].ref;
                    bestRefToFrame = potentialReferenceFrames[i].refToFrame;
                    bestRefToFrame_tracked = RefToFrame_tracked;
                    bestDist = dist.dot(dist);
                    bestUsage = tracker->pointUsage;
                }
            }
        }

        // ***************** 如果找到了最新关键帧frame的最佳参考帧，就返回该参考帧，否则返回0  *****************
        if (bestFrame != 0)
        {
            if (enablePrintDebugInfo && printRelocalizationInfo)
                printf("FindReferences for %d: Checked %d (%d). dist %.3f + usage %.3f = %.3f. pose discrepancy %.2f. TAKE %d!\n",
                       (int)frame->id(), (int)potentialReferenceFrames.size(), checkedSecondary, bestDist, bestUsage, bestScore,
                       bestPoseDiscrepancy, bestFrame->id());
            return bestFrame;
        }
        else
        {
            if (enablePrintDebugInfo && printRelocalizationInfo)
                printf("FindReferences for %d: Checked %d (%d), bestScore %.2f. MAKE NEW\n", (int)frame->id(),
                       (int)potentialReferenceFrames.size(), checkedSecondary, bestScore);
            return 0;
        }
    }

    // ! 在已经加入g2o的关键帧中找到，当前关键帧可以追踪到的其他关键帧，作为候选帧
    std::unordered_set<Frame *, std::hash<Frame *>, std::equal_to<Frame *>, Eigen::aligned_allocator<Frame *>>
    TrackableKeyFrameSearch::findCandidates(Frame *keyframe, Frame *&fabMapResult_out, bool includeFABMAP, bool closenessTH)
    {   
        /* 
            返回值是一个无序哈希表unordered_set
            std::equal<> 允许将两个元素传递给它，然后返回bool值，表明这两个元素是否相等
            Eigen::aligned_allocator<>  是 Eigen 库中定义的一个分配器类，用于分配内存并确保分配的内存满足特定的对齐要求。
        */
        std::unordered_set<Frame *, std::hash<Frame *>, std::equal_to<Frame *>, Eigen::aligned_allocator<Frame *>> results;

        // ***************** 寻找当前关键帧所有潜在的参考帧 *****************
        // Add all candidates that are similar in an euclidean sense.
        std::vector<TrackableKFStruct, Eigen::aligned_allocator<TrackableKFStruct>> potentialReferenceFrames =
            findEuclideanOverlapFrames(keyframe, closenessTH * 15 / (KFDistWeight * KFDistWeight), 1.0 - 0.25 * closenessTH,
                                       true);
        // 将潜在的参考帧存入变量results中
        for (unsigned int i = 0; i < potentialReferenceFrames.size(); i++)
            results.insert(potentialReferenceFrames[i].ref);

        int appearanceBased = 0;
        fabMapResult_out = 0;
        // * 是否使用FAB-MAP回环检测
        if (includeFABMAP)
        {
            // Add Appearance-based Candidate, and all it's neighbours.
            fabMapResult_out = findAppearanceBasedCandidate(keyframe);
            if (fabMapResult_out != nullptr)
            {
                results.insert(fabMapResult_out);
                results.insert(fabMapResult_out->neighbors.begin(), fabMapResult_out->neighbors.end());
                appearanceBased = 1 + fabMapResult_out->neighbors.size();
            }
        }
        // 输出debug信息
        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf("Early LoopClosure-Candidates for %d: %d euclidean, %d appearance-based, %d total\n", (int)keyframe->id(),
                   (int)potentialReferenceFrames.size(), appearanceBased, (int)results.size());

        return results;
    }

    Frame *TrackableKeyFrameSearch::findAppearanceBasedCandidate(Frame *keyframe)
    {
#ifdef HAVE_FABMAP
        if (!useFabMap)
            return nullptr;

        if (!fabMap.isValid())
        {
            printf("Error: called findAppearanceBasedCandidate(), but FabMap instance is not valid!\n");
            return nullptr;
        }

        int newID, loopID;
        fabMap.compareAndAdd(keyframe, &newID, &loopID);
        if (newID < 0)
            return nullptr;

        fabmapIDToKeyframe.insert(std::make_pair(newID, keyframe));
        if (loopID >= 0)
            return fabmapIDToKeyframe.at(loopID);
        else
            return nullptr;
#else
        if (useFabMap)
            printf("Warning: Compiled without FabMap, but useFabMap is enabled... ignoring.\n");
        return nullptr;
#endif
    }

} // namespace lsd_slam
