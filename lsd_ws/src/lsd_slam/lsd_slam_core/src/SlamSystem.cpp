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

#include "SlamSystem.h"

#include "DataStructures/Frame.h"
#include "Tracking/SE3Tracker.h"
#include "Tracking/Sim3Tracker.h"
#include "DepthEstimation/DepthMap.h"
#include "Tracking/TrackingReference.h"
#include "LiveSLAMWrapper.h"
#include "util/globalFuncs.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "GlobalMapping/TrackableKeyFrameSearch.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include <g2o/core/robust_kernel_impl.h>
#include "DataStructures/FrameMemory.h"
#include "deque"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>

#ifdef ANDROID
#include <android/log.h>
#endif

#include "opencv4/opencv2/opencv.hpp"

using namespace lsd_slam;

// 创建slam系统
SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool enableSLAM) : SLAMEnabled(enableSLAM), relocalizer(w, h, K)
{
    if (w % 16 != 0 || h % 16 != 0)
    {
        printf("image dimensions must be multiples of 16! Please crop your images / video accordingly.\n");
        assert(false);
    }

    // *************** 读取相机参数 ***************
    this->width = w;
    this->height = h;
    this->K = K;
    trackingIsGood = true;

    // *************** 构建一个新的g2o位姿图 ***************
    currentKeyFrame = nullptr;                  // 当前帧
    trackingReferenceFrameSharedPT = nullptr;
    keyFrameGraph = new KeyFrameGraph();
    createNewKeyFrame = false;

    // *************** 创建深度图 ***************
    map = new DepthMap(w, h, K);

    newConstraintAdded = false;
    haveUnmergedOptimizationOffset = false;

    // *************** 创建追踪器 ***************
    tracker = new SE3Tracker(w, h, K);
    // Do not use more than 4 levels for odometry tracking，这里的PYRAMID_LEVELS=5
    for (int level = 4; level < PYRAMID_LEVELS; ++level)
        tracker->settings.maxItsPerLvl[level] = 0;

    // *************** 创建追踪时用到的参考帧 ***************
    trackingReference = new TrackingReference();
    mappingTrackingReference = new TrackingReference();

    if (SLAMEnabled) // setting.cpp中SLAMEnabled默认为true
    {
        trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph, w, h, K);
        constraintTracker = new Sim3Tracker(w, h, K);
        constraintSE3Tracker = new SE3Tracker(w, h, K);
        newKFTrackingReference = new TrackingReference();
        candidateTrackingReference = new TrackingReference();
    }
    else
    {
        constraintSE3Tracker = 0;
        trackableKeyFrameSearch = 0;
        constraintTracker = 0;
        newKFTrackingReference = 0;
        candidateTrackingReference = 0;
    }

    outputWrapper = 0;

    keepRunning = true; //  下面的建图线程thread_mapping中用到
    doFinalOptimization = false;
    depthMapScreenshotFlag = false;
    lastTrackingClosenessScore = 0;

    // *************** 子线程1：创建建图线程 ***************
    thread_mapping = boost::thread(&SlamSystem::mappingThreadLoop, this);

    if (SLAMEnabled)
    {   
        // 建图一致性约束也就是做闭环检测和全局优化
        // *************** 子线程2：创建查找一致性约束的线程 ***************
        thread_constraint_search = boost::thread(&SlamSystem::constraintSearchThreadLoop, this);
        // *************** 子线程3：创建全局优化的线程 ***************
        thread_optimization = boost::thread(&SlamSystem::optimizationThreadLoop, this);
    }

    msTrackFrame = msOptimizationIteration = msFindConstraintsItaration = msFindReferences = 0;
    nTrackFrame = nOptimizationIteration = nFindConstraintsItaration = nFindReferences = 0;
    nAvgTrackFrame = nAvgOptimizationIteration = nAvgFindConstraintsItaration = nAvgFindReferences = 0;

    // 得到线程启动到当前的时间，精确到ms
    gettimeofday(&lastHzUpdate, NULL);
}

SlamSystem::~SlamSystem()
{
    keepRunning = false;

    // make sure none is waiting for something.
    printf("... waiting for SlamSystem's threads to exit\n");
    // xx.notify_all()：唤醒所有在等待xx条件变量的线程
    newFrameMappedSignal.notify_all();
    unmappedTrackedFramesSignal.notify_all();
    newKeyFrameCreatedSignal.notify_all();
    newConstraintCreatedSignal.notify_all();

    thread_mapping.join();
    thread_constraint_search.join();
    thread_optimization.join();
    printf("DONE waiting for SlamSystem's threads to exit\n");

    if (trackableKeyFrameSearch != 0)
        delete trackableKeyFrameSearch;
    if (constraintTracker != 0)
        delete constraintTracker;
    if (constraintSE3Tracker != 0)
        delete constraintSE3Tracker;
    if (newKFTrackingReference != 0)
        delete newKFTrackingReference;
    if (candidateTrackingReference != 0)
        delete candidateTrackingReference;

    delete mappingTrackingReference;
    delete map;
    delete trackingReference;
    delete tracker;

    // make shure to reset all shared pointers to all frames before deleting the keyframegraph!
    unmappedTrackedFrames.clear();      // 清空保存的图像帧队列（在trackframe中结束时每一帧都备份在这个变量）
    latestFrameTriedForReloc.reset();
    latestTrackedFrame.reset();
    currentKeyFrame.reset();
    trackingReferenceFrameSharedPT.reset();

    // delte keyframe graph
    delete keyFrameGraph;

    FrameMemory::getInstance().releaseBuffes();

    Util::closeAllWindows();
}

void SlamSystem::setVisualization(Output3DWrapper *outputWrapper)
{
    this->outputWrapper = outputWrapper;
}

// ! 将关键帧的位姿更新为优化完后的位姿, 只有在优化进程优化过了后才会更新
void SlamSystem::mergeOptimizationOffset()
{
    // update all vertices that are in the graph!
    // 这里会和回环线程产生互斥
    poseConsistencyMutex.lock();

    bool needPublish = false;
    // ? haveUnmergedOptimizationOffset: 默认为false，只有在优化线程开始优化迭代后，才为true
    if (haveUnmergedOptimizationOffset)
    {
        // * 将关键帧的位姿更新为优化完后的位姿
        keyFrameGraph->keyframesAllMutex.lock_shared();
        for (unsigned int i = 0; i < keyFrameGraph->keyframesAll.size(); i++)
            keyFrameGraph->keyframesAll[i]->pose->applyPoseGraphOptResult();
        keyFrameGraph->keyframesAllMutex.unlock_shared();

        haveUnmergedOptimizationOffset = false;
        needPublish = true;
    }

    poseConsistencyMutex.unlock();

    if (needPublish)
        publishKeyframeGraph();
}

// ! 建图线程
void SlamSystem::mappingThreadLoop()
{
    printf("Started mapping thread!\n");
    // 只要slam系统在跑，就一直建图
    while (keepRunning)
    {   
        // ************* 建图，成功就返回true *************
        if (!doMappingIteration())
        {   
            // 锁住名为unmappedTrackedFramesMutex的锁
            boost::unique_lock<boost::mutex> lock(unmappedTrackedFramesMutex);

            // * 如果对某一帧建图失败，需要等待跟踪线程的唤醒
            // 获得unmappedTrackedFramesMutex的锁的基础上，等待条件变量unmappedTrackedFramesSignal被通知
            // 最多等待200ms
            unmappedTrackedFramesSignal.timed_wait(lock, boost::posix_time::milliseconds(200)); // slight chance of deadlock
                                                                                                // otherwise
            // 解锁
            lock.unlock();
        }

        newFrameMappedMutex.lock();
        newFrameMappedSignal.notify_all();  // 唤醒所有等待在 newFrameMappedSignal 条件变量上的线程。
        newFrameMappedMutex.unlock();
    }
    printf("Exited mapping thread \n");
}

void SlamSystem::finalize()
{
    printf("Finalizing Graph... finding final constraints!!\n");

    lastNumConstraintsAddedOnFullRetrack = 1;
    while (lastNumConstraintsAddedOnFullRetrack != 0)
    {
        doFullReConstraintTrack = true;
        usleep(200000);
    }

    printf("Finalizing Graph... optimizing!!\n");
    doFinalOptimization = true;
    newConstraintMutex.lock();
    newConstraintAdded = true;
    newConstraintCreatedSignal.notify_all();
    newConstraintMutex.unlock();
    while (doFinalOptimization)
    {
        usleep(200000);
    }

    printf("Finalizing Graph... publishing!!\n");
    // 如果跟踪失败没有建图，建图进程会被unmappedTrackedFramesSignal条件变量卡住，这里是让建图进程继续下去
    unmappedTrackedFramesMutex.lock();
    unmappedTrackedFramesSignal.notify_one();
    unmappedTrackedFramesMutex.unlock();
    while (doFinalOptimization)
    {
        usleep(200000);
    }
    boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
    newFrameMappedSignal.wait(lock);
    newFrameMappedSignal.wait(lock);

    usleep(200000);
    printf("Done Finalizing Graph.!!\n");
}

// ! 一致性约束线程（回环跟踪））
void SlamSystem::constraintSearchThreadLoop()
{
    printf("Started  constraint search thread!\n");
    
    // 创建一个独占锁（unique lock），并用这个锁锁住名为 newKeyFrameMutex 的互斥量。
    // 控制建图线程在调用finishCurrentKeyframe()时修改newKeyFrames
    boost::unique_lock<boost::mutex> lock(newKeyFrameMutex);
    int failedToRetrack = 0;

    while (keepRunning)
    {   
        // ************* 1. 如果新关键帧队列newKeyFrames为空，则在所有关键帧中随机选取测试闭环 *************
        // 系统刚运行的时候，newKeyFrames一开始也是空的
        // 建图线程会在为新关键帧建图时调用finishCurrentKeyframe()来建图，将新关键帧插入这个双端队列newKeyFrames
        if (newKeyFrames.size() == 0)
        {
            lock.unlock();
            keyFrameGraph->keyframesForRetrackMutex.lock();
            bool doneSomething = false;

            // keyframesForRetrack：g2o图中包含所有的关键帧，如果超过10张就尝试回环
            if (keyFrameGraph->keyframesForRetrack.size() > 10)
            {
                // * 用std::deque的迭代器从已经加入g2o的关键帧中随机选取一个关键帧
                // 从双端队列keyFrameGraph->keyframesForRetrack的前三个关键帧中选一个，并将它的地址给toReTrackFrame
                std::deque<Frame *>::iterator toReTrack =
                    keyFrameGraph->keyframesForRetrack.begin() + (rand() % (keyFrameGraph->keyframesForRetrack.size() / 3));
                Frame *toReTrackFrame = *toReTrack;

                // * 将选择出来的关键帧放到双端队列keyframesForRetrack的末尾
                keyFrameGraph->keyframesForRetrack.erase(toReTrack);
                keyFrameGraph->keyframesForRetrack.push_back(toReTrackFrame);

                keyFrameGraph->keyframesForRetrackMutex.unlock();

                // * 
                // 第一个false：如果上一个关键帧和当前关键帧之间的相对位姿很小，不为它再做一遍回环跟踪直接返回。
                // 第二个false: 说明不用fabmap
                int found = findConstraintsForNewKeyFrames(toReTrackFrame, false, false, 2.0);
                if (found == 0)
                    failedToRetrack++;
                else
                    failedToRetrack = 0;

                // 在一开始由于没有足够的帧，所以都是false
                // 之后如果回环线程没有出错，donesomething都是true
                if (failedToRetrack < (int)keyFrameGraph->keyframesForRetrack.size() - 5)
                    doneSomething = true;
            }
            else
                keyFrameGraph->keyframesForRetrackMutex.unlock();

            lock.lock();

            // * 在系统刚开始，没有什么帧可以参与回环，所以会卡在这，等待建图线程添加完第一个关键帧后唤醒
            if (!doneSomething)
            {
                if (enablePrintDebugInfo && printConstraintSearchInfo)
                    printf("nothing to re-track... waiting.\n");
                newKeyFrameCreatedSignal.timed_wait(lock, boost::posix_time::milliseconds(500));
            }
        }
        // ************* 2. 如果新关键帧队列不为空，则取最早的新关键帧测试闭环*************
        else
        {   
            // 读取当前最老的关键帧
            Frame *newKF = newKeyFrames.front();
            newKeyFrames.pop_front();
            lock.unlock();

            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);

            // * 
            // 第一个true： 如果上一个关键帧和当前关键帧之间的相对位姿很小，还是为它再做一遍回环跟踪直接返回。
            // 第二个true: 说明用fabmap来检测回环
            findConstraintsForNewKeyFrames(newKF, true, true, 1.0);
            failedToRetrack = 0;
            gettimeofday(&tv_end, NULL);
            msFindConstraintsItaration =
                0.9 * msFindConstraintsItaration +
                0.1 * ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
            nFindConstraintsItaration++;

            FrameMemory::getInstance().pruneActiveFrames();
            lock.lock();
        }

        // ************* 3. 对所有的关键帧进行回环跟踪 *************
        if (doFullReConstraintTrack)
        {
            lock.unlock();
            printf("Optizing Full Map!\n");

            int added = 0;
            for (unsigned int i = 0; i < keyFrameGraph->keyframesAll.size(); i++)
            {
                if (keyFrameGraph->keyframesAll[i]->pose->isInGraph)
                    // 第一个false：如果上一个关键帧和当前关键帧之间的相对位姿很小，不为它再做一遍回环跟踪直接返回。
                    // 第二个false: 说明不用fabmap
                    added += findConstraintsForNewKeyFrames(keyFrameGraph->keyframesAll[i], false, false, 1.0);
            }

            printf("Done optizing Full Map! Added %d constraints.\n", added);

            doFullReConstraintTrack = false;

            lastNumConstraintsAddedOnFullRetrack = added;
            lock.lock();
        }
    }

    printf("Exited constraint search thread \n");
}

// ! 全局优化线程
void SlamSystem::optimizationThreadLoop()
{
    printf("Started optimization thread \n");

    while (keepRunning)
    {
        boost::unique_lock<boost::mutex> lock(newConstraintMutex);
        if (!newConstraintAdded)
            newConstraintCreatedSignal.timed_wait(lock, boost::posix_time::milliseconds(2000)); // slight chance of deadlock
                                                                                                // otherwise
        newConstraintAdded = false;
        lock.unlock();

        if (doFinalOptimization)
        {
            printf("doing final optimization iteration!\n");
            optimizationIteration(50, 0.001);
            doFinalOptimization = false;
        }
        while (optimizationIteration(5, 0.02))
            ;
    }

    printf("Exited optimization thread \n");
}

void SlamSystem::publishKeyframeGraph()
{
    if (outputWrapper != nullptr)
        outputWrapper->publishKeyframeGraph(keyFrameGraph);
}

void SlamSystem::requestDepthMapScreenshot(const std::string &filename)
{
    depthMapScreenshotFilename = filename;
    depthMapScreenshotFlag = true;
}

// ! 每当构造完一个关键帧都会调用，对当前关键帧的深度图进行一次填补，并计算它的平均深度
// 对于第一个关键帧(也就是第一帧)还会将它加入到g2o图中去
void SlamSystem::finishCurrentKeyframe()
{
    if (enablePrintDebugInfo && printThreadingInfo)
        printf("FINALIZING KF %d\n", currentKeyFrame->id());

    // * 对当前关键帧的深度图进行一次填补，并计算它的平均深度
    map->finalizeKeyFrame();

    if (SLAMEnabled)
    {
        // * 将当前帧设为最新的关键帧（当前正在追踪的参考帧）
        // 将最新的关键帧设为：建图时当前帧正在追踪的参考帧(最新的关键帧)
        mappingTrackingReference->importFrame(currentKeyFrame.get());
        // Frame::currentKeyFrame保存着所有关键帧各种信息(颜色，方差，位姿)
        currentKeyFrame->setPermaRef(mappingTrackingReference);
        // 将加载新关键帧时激活的互斥锁 解锁
        mappingTrackingReference->invalidate();

        // * 对于第一个关键帧(也就是第一帧)还会将它加入到g2o图中去
        // idxInKeyframes在Frame被初始化时是-1，如果<0说明现在关键帧的g2o图里还没有节点
        if (currentKeyFrame->idxInKeyframes < 0)
        {   
            keyFrameGraph->keyframesAllMutex.lock();
            // 因为size()返回个数，序号从0开始比size小1，所以size()可以直接是新帧的索引
            currentKeyFrame->idxInKeyframes = keyFrameGraph->keyframesAll.size();
            keyFrameGraph->keyframesAll.push_back(currentKeyFrame.get());
            keyFrameGraph->totalPoints += currentKeyFrame->numPoints;
            keyFrameGraph->totalVertices++;
            keyFrameGraph->keyframesAllMutex.unlock();

            // * 告诉一致性约束线程新的关键帧被加入了，可以进入下一个循环了
            newKeyFrameMutex.lock();
            // 把新创建的关键帧插入双端队列std::deque的尾部
            newKeyFrames.push_back(currentKeyFrame.get());
            newKeyFrameCreatedSignal.notify_all();
            newKeyFrameMutex.unlock();
        }
    }

    if (outputWrapper != 0)
        outputWrapper->publishKeyframe(currentKeyFrame.get());
}

// ! 把该关键帧直接从keyFrameGraph中剔除。
// 在跟踪线程trackFrame中，每一帧图像都加入了图keyFrameGraph（这个有点像关键帧候选队列）
void SlamSystem::discardCurrentKeyframe()
{
    if (enablePrintDebugInfo && printThreadingInfo)
        printf("DISCARDING KF %d\n", currentKeyFrame->id());

    // * idxInKeyframes在Frame被初始化时是-1，如果>=0说明当前关键帧已经被加入到图优化g2o中了，不再删除改关键帧
    if (currentKeyFrame->idxInKeyframes >= 0)
    {
        printf("WARNING: trying to discard a KF that has already been added to the graph... finalizing instead.\n");
        finishCurrentKeyframe();
        return;
    }

    // * 将该关键帧的建图无效化
    map->invalidate();

    // * 将该关键帧从图优化g2o中删除
    keyFrameGraph->allFramePosesMutex.lock();
    for (FramePoseStruct *p : keyFrameGraph->allFramePoses)
    {
        if (p->trackingParent != 0 && p->trackingParent->frameID == currentKeyFrame->id())
            p->trackingParent = 0;
    }
    keyFrameGraph->allFramePosesMutex.unlock();

    keyFrameGraph->idToKeyFrameMutex.lock();
    keyFrameGraph->idToKeyFrame.erase(currentKeyFrame->id());
    keyFrameGraph->idToKeyFrameMutex.unlock();
}

// ! 构建新的当前关键帧，用参考帧传播并构建新关键帧的深度图
void SlamSystem::createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate)
{
    if (enablePrintDebugInfo && printThreadingInfo)
        printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate->id(), currentKeyFrame->id());

    if (SLAMEnabled)
    {
        // add NEW keyframe to id-lookup
        keyFrameGraph->idToKeyFrameMutex.lock();
        // 把当前用于构建新关键帧的帧放入关键帧队列以及传播逆深度
        keyFrameGraph->idToKeyFrame.insert(std::make_pair(newKeyframeCandidate->id(), newKeyframeCandidate));
        keyFrameGraph->idToKeyFrameMutex.unlock();
    }

    // * 传播并构建新关键帧的深度图
    // propagate & make new.
    map->createKeyFrame(newKeyframeCandidate.get());

    if (printPropagationStatistics)
    {
        Eigen::Matrix<float, 20, 1> data;
        data.setZero();
        data[0] = runningStats.num_prop_attempts / ((float)width * height);
        data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
        data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

        outputWrapper->publishDebugInfo(data);
    }

    currentKeyFrameMutex.lock();
    currentKeyFrame = newKeyframeCandidate;
    currentKeyFrameMutex.unlock();
}

// ! 将关键帧keyframeToLoad重新设为还需要继续优化的状态，也就是还是当前的参考帧
void SlamSystem::loadNewCurrentKeyframe(Frame *keyframeToLoad)
{
    if (enablePrintDebugInfo && printThreadingInfo)
        printf("RE-ACTIVATE KF %d\n", keyframeToLoad->id());

    // 将关键帧keyframeToLoad的地图重新设置为激活状态(可修改需要优化的)
    map->setFromExistingKF(keyframeToLoad);

    if (enablePrintDebugInfo && printRegularizeStatistics)
        printf("re-activate frame %d!\n", keyframeToLoad->id());

    // 将当前的参考帧(最新的关键帧)设为keyframeToLoad
    currentKeyFrameMutex.lock();
    currentKeyFrame = keyFrameGraph->idToKeyFrame.find(keyframeToLoad->id())->second;
    currentKeyFrame->depthHasBeenUpdatedFlag = false;
    currentKeyFrameMutex.unlock();
}

// ! 改变当前的关键帧：currentKeyFrame，如果在地图中存在与当前候选关键帧很相似的关键帧，则使用地图中已有的关键帧，否则重新构建关键帧并为其建图进行深度图传播。
void SlamSystem::changeKeyframe(bool noCreate, bool force, float maxScore)
{
    Frame *newReferenceKF = 0;
    // latestTrackedFrame是当前正在处理的帧
    std::shared_ptr<Frame> newKeyframeCandidate = latestTrackedFrame;
    
    // *************** 1. 在已经加入到图优化g2o中的关键帧中，寻找新关键帧frame的所有能跟踪到的作为它的参考帧，并计算该最新关键帧的分数(公式16) ***************
    // setting.cpp中都默认为true
    if (doKFReActivation && SLAMEnabled)
    {
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);
        // * 如果分数不够高(相机移动距离不够远)就要在这些能追踪到的参考帧里选一个作为当前关键帧的替代，因为论文中要求只有足够远才能成为关键帧
        newReferenceKF = trackableKeyFrameSearch->findRePositionCandidate(newKeyframeCandidate.get(), maxScore);
        gettimeofday(&tv_end, NULL);
        msFindReferences = 0.9 * msFindReferences + 0.1 * ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                           (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        nFindReferences++;
    }

    if (newReferenceKF != 0)
        // * 如果在已加入g2o的所有旧关键帧中找到了距离不够远的关键帧，当前帧就不能被设置为关键帧了，那个被找到的关键帧会被重新激活继续作为当前关键帧来使用
        loadNewCurrentKeyframe(newReferenceKF);

    // *************** 2. 如果当前帧距离最近的关键帧都移动了足够远的距离 ***************
    else
    {
        if (force)
        {   
            // * 要么重定位当前帧
            if (noCreate)
            {
                trackingIsGood = false;
                nextRelocIdx = -1;
                printf("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
            }
            // * 要么强制将其创建为新的关键帧，并为其重新构建关键帧并为其建图进行深度图传播
            else
                createNewCurrentKeyframe(newKeyframeCandidate);
        }
    }

    createNewKeyFrame = false;
}

// ! 用当前所有保存的能跟踪到参考帧的普通帧来更新它的参考帧(最新的关键帧)的深度
bool SlamSystem::updateKeyframe()
{
    std::shared_ptr<Frame> reference = nullptr;
    std::deque<std::shared_ptr<Frame>> references;

    unmappedTrackedFramesMutex.lock();
    
    // *************** 1. 把所有不是跟踪到当前关键帧的图像帧都从队列中剔除 ***************
    // remove frames that have a different tracking parent.
    while (unmappedTrackedFrames.size() > 0 &&
           (!unmappedTrackedFrames.front()->hasTrackingParent() ||
            unmappedTrackedFrames.front()->getTrackingParent() != currentKeyFrame.get()))
    {
        unmappedTrackedFrames.front()->clear_refPixelWasGood();
        unmappedTrackedFrames.pop_front();
    }

    // *************** 2. 克隆保存的图像帧队列（在trackframe中结束时每一帧都备份在这个变量）到reference中去 ***************
    // clone list
    if (unmappedTrackedFrames.size() > 0)
    {
        for (unsigned int i = 0; i < unmappedTrackedFrames.size(); i++)
            references.push_back(unmappedTrackedFrames[i]);

        std::shared_ptr<Frame> popped = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();
        unmappedTrackedFramesMutex.unlock();

        if (enablePrintDebugInfo && printThreadingInfo)
            printf("MAPPING %d on %d to %d (%d frames)\n", currentKeyFrame->id(), references.front()->id(),
                   references.back()->id(), (int)references.size());

        // * 2.1 用当前所有未建图过的普通帧来更新当前关键帧的深度图
        map->updateKeyframe(references);

        popped->clear_refPixelWasGood();
        // 清空所有保存的普通帧
        references.clear();
    }
    else
    {   
        // * 2.2 如果当前关键帧没有可以用来更新自己深度图的普通帧就返回false，导致当前帧建图失败
        unmappedTrackedFramesMutex.unlock();
        return false;
    }

    if (enablePrintDebugInfo && printRegularizeStatistics)
    {
        Eigen::Matrix<float, 20, 1> data;
        data.setZero();
        data[0] = runningStats.num_reg_created;
        data[2] = runningStats.num_reg_smeared;
        data[3] = runningStats.num_reg_deleted_secondary;
        data[4] = runningStats.num_reg_deleted_occluded;
        data[5] = runningStats.num_reg_blacklisted;

        data[6] = runningStats.num_observe_created;
        data[7] = runningStats.num_observe_create_attempted;
        data[8] = runningStats.num_observe_updated;
        data[9] = runningStats.num_observe_update_attempted;

        data[10] = runningStats.num_observe_good;
        data[11] = runningStats.num_observe_inconsistent;
        data[12] = runningStats.num_observe_notfound;
        data[13] = runningStats.num_observe_skip_oob;
        data[14] = runningStats.num_observe_skip_fail;

        outputWrapper->publishDebugInfo(data);
    }

    if (outputWrapper != 0 && continuousPCOutput && currentKeyFrame != 0)
        outputWrapper->publishKeyframe(currentKeyFrame.get());

    return true;
}

// ! 进行时间采样，并计算一些与时间相关的性能指标
void SlamSystem::addTimingSamples()
{
    map->addTimingSample();
    struct timeval now;
    gettimeofday(&now, NULL);
    float sPassed = ((now.tv_sec - lastHzUpdate.tv_sec) + (now.tv_usec - lastHzUpdate.tv_usec) / 1000000.0f);
    if (sPassed > 1.0f)
    {
        nAvgTrackFrame = 0.8 * nAvgTrackFrame + 0.2 * (nTrackFrame / sPassed);
        nTrackFrame = 0;
        nAvgOptimizationIteration = 0.8 * nAvgOptimizationIteration + 0.2 * (nOptimizationIteration / sPassed);
        nOptimizationIteration = 0;
        nAvgFindReferences = 0.8 * nAvgFindReferences + 0.2 * (nFindReferences / sPassed);
        nFindReferences = 0;

        if (trackableKeyFrameSearch != 0)
        {
            trackableKeyFrameSearch->nAvgTrackPermaRef =
                0.8 * trackableKeyFrameSearch->nAvgTrackPermaRef + 0.2 * (trackableKeyFrameSearch->nTrackPermaRef / sPassed);
            trackableKeyFrameSearch->nTrackPermaRef = 0;
        }
        nAvgFindConstraintsItaration = 0.8 * nAvgFindConstraintsItaration + 0.2 * (nFindConstraintsItaration / sPassed);
        nFindConstraintsItaration = 0;
        nAvgOptimizationIteration = 0.8 * nAvgOptimizationIteration + 0.2 * (nOptimizationIteration / sPassed);
        nOptimizationIteration = 0;

        lastHzUpdate = now;

        if (enablePrintDebugInfo && printOverallTiming)
        {
            printf("MapIt: %3.1fms (%.1fHz); Track: %3.1fms (%.1fHz); Create: %3.1fms (%.1fHz); FindRef: %3.1fms (%.1fHz); "
                   "PermaTrk: %3.1fms (%.1fHz); Opt: %3.1fms (%.1fHz); FindConst: %3.1fms (%.1fHz);\n",
                   map->msUpdate, map->nAvgUpdate, msTrackFrame, nAvgTrackFrame, map->msCreate + map->msFinalize,
                   map->nAvgCreate, msFindReferences, nAvgFindReferences,
                   trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->msTrackPermaRef : 0,
                   trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->nAvgTrackPermaRef : 0, msOptimizationIteration,
                   nAvgOptimizationIteration, msFindConstraintsItaration, nAvgFindConstraintsItaration);
        }
    }
}

void SlamSystem::debugDisplayDepthMap()
{
    map->debugPlotDepthMap();
    double scale = 1;
    if (currentKeyFrame != 0 && currentKeyFrame != 0)
        scale = currentKeyFrame->getScaledCamToWorld().scale();
    // debug plot depthmap
    char buf1[200];
    char buf2[200];

    snprintf(buf1, 200, "Map: Upd %3.0fms (%2.0fHz); Trk %3.0fms (%2.0fHz); %d / %d / %d", map->msUpdate, map->nAvgUpdate,
             msTrackFrame, nAvgTrackFrame, currentKeyFrame->numFramesTrackedOnThis, currentKeyFrame->numMappedOnThis,
             (int)unmappedTrackedFrames.size());

    snprintf(buf2, 200,
             "dens %2.0f%%; good %2.0f%%; scale %2.2f; res %2.1f/; usg %2.0f%%; Map: %d F, %d KF, %d E, %.1fm Pts",
             100 * currentKeyFrame->numPoints / (float)(width * height), 100 * tracking_lastGoodPerBad, scale,
             tracking_lastResidual, 100 * tracking_lastUsage, (int)keyFrameGraph->allFramePoses.size(),
             keyFrameGraph->totalVertices, (int)keyFrameGraph->edgesAll.size(), 1e-6 * (float)keyFrameGraph->totalPoints);

    if (onSceenInfoDisplay)
        printMessageOnCVImage(map->debugImageDepth, buf1, buf2);
    if (displayDepthMap)
        Util::displayImage("DebugWindow DEPTH", map->debugImageDepth, false);

    int pressedKey = Util::waitKey(1);
    handleKey(pressedKey);
}

void SlamSystem::takeRelocalizeResult()
{
    Frame *keyframe;
    int succFrameID;
    SE3 succFrameToKF_init;
    std::shared_ptr<Frame> succFrame;
    relocalizer.stop();         // 结束重定位
    relocalizer.getResult(keyframe, succFrame, succFrameID, succFrameToKF_init);
    assert(keyframe != 0);

    loadNewCurrentKeyframe(keyframe);

    currentKeyFrameMutex.lock();
    trackingReference->importFrame(currentKeyFrame.get());
    trackingReferenceFrameSharedPT = currentKeyFrame;
    currentKeyFrameMutex.unlock();

    tracker->trackFrame(trackingReference, succFrame.get(), succFrameToKF_init);

    if (!tracker->trackingWasGood || tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount) <
                                         1 - 0.75f * (1 - MIN_GOODPERGOODBAD_PIXEL))
    {
        if (enablePrintDebugInfo && printRelocalizationInfo)
            printf("RELOCALIZATION FAILED BADLY! discarding result.\n");
        trackingReference->invalidate();
    }
    else
    {
        keyFrameGraph->addFrame(succFrame.get());

        unmappedTrackedFramesMutex.lock();
        if (unmappedTrackedFrames.size() < 50)
            unmappedTrackedFrames.push_back(succFrame);
        unmappedTrackedFramesMutex.unlock();

        currentKeyFrameMutex.lock();
        createNewKeyFrame = false;
        trackingIsGood = true;
        currentKeyFrameMutex.unlock();
    }
}

// ! 建图，只对关键帧进行建图
bool SlamSystem::doMappingIteration()
{
    // 如果等于0，说明跟踪主线程还没有进行初始化，直接返回false
    if (currentKeyFrame == 0)
        return false;

    // *************** 1. idxInKeyframes在Frame被初始化时是-1，<0表示当前g2o图中没有任何关键帧 ***************
    // doMapping在setting.cpp中默认为true，如果人为设置为false整个建图线程没啥用了。
    // 所以!doMapping 貌似是false，那下面这段就没啥用了？
    if (!doMapping && currentKeyFrame->idxInKeyframes < 0)
    {   
        // * 决定是否把当前关键帧加入
        // MIN_NUM_MAPPED在setting.cpp中是5
        // numMappedOnThisTotal：根据当前参考帧(即最新的关键帧)来计算相对位姿的帧的数量, 即跟踪到当前关键帧的帧数
        // 要求每个关键帧都必须至少要有5个普通帧的位姿是根据这个关键帧算的
        if (currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
            // 对当前关键帧的深度图进行一次填补，并计算它的平均深度
            finishCurrentKeyframe();
        else
            // 把该关键帧直接从keyFrameGraph中剔除。
            discardCurrentKeyframe();
        
        // * 因为!doMapping所以当前帧不建图
        map->invalidate();
        printf("Finished KF %d as Mapping got disabled!\n", currentKeyFrame->id());

        // * 决定当前帧是否应该成为新的关键帧
        // 两个true表示，如果当前帧在自己参考帧的足够距离之外了，设置trackingIsGood = false;说明当前帧不够好不能被设为关键帧
        // 如果当前帧在自己参考帧的足够距离之内，自然那个参考帧还是当前最新的关键帧
        changeKeyframe(true, true, 1.0f);
    }

    // *************** 2. 更新后端优化的结果 ***************
    // 将关键帧的位姿更新为优化完后的位姿, 只有优化进程
    mergeOptimizationOffset();
    // 进行时间采样，并计算一些与时间相关的性能指标
    addTimingSamples();

    // 转存地图，默认为false不存
    if (dumpMap)
    {
        keyFrameGraph->dumpMap(packagePath + "/save");
        dumpMap = false;
    }

    // *************** 3. 正式建图 ***************
    // set mappingFrame
    // * 3.1 当前帧跟踪的好
    if (trackingIsGood)
    {
        // doMapping默认为true，下面这段所以没啥用
        if (!doMapping)
        {
            // printf("tryToChange refframe, lastScore %f!\n", lastTrackingClosenessScore);
            if (lastTrackingClosenessScore > 1)
                changeKeyframe(true, false, lastTrackingClosenessScore * 0.75);

            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();

            return false;
        }

        // * 如果当前帧是关键帧，就为它构建自己的深度图
        // createNewKeyFrame默认为false，在跟踪线程中(bookmark：新关键帧1)被设置为true
        if (createNewKeyFrame)
        {   
            // 对当前关键帧的深度图进行一次填补，并计算它的平均深度
            finishCurrentKeyframe();
            // 将当前帧设置为关键帧
            // true表示如果当前关键帧移动的距离足够远就执行：false表示不重定位而是将其创建为关键帧
            changeKeyframe(false, true, 1.0f);

            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();
        }
        // * 如果不是关键帧，就用当前所有保存的能跟踪到参考帧的普通帧来更新它的参考帧(最新的关键帧)的深度
        else
        {   
            // 用当前所有保存的能跟踪到参考帧的普通帧来更新它的参考帧(最新的关键帧)的深度
            // 这些普通帧都保存在unmappedTrackedFrames变量中
            bool didSomething = updateKeyframe();

            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();
            
            // 如果当前关键帧没有可以用来更新自己深度图的普通帧
            if (!didSomething)
                return false;
        }

        return true;
    }
    // * 3.2 跟踪不好时需要对当前帧进行重定位，如果正在给最新关键帧建图就对其深度图更新后关闭
    else
    {
        // * 如果正在给最新关键帧建图就对其深度图更新后关闭
        // invalidate map if it was valid.
        // 如果当前地图有活跃的关键帧，即activeKeyFrame！=0
        if (map->isValid())
        {
            // MIN_NUM_MAPPED=5，如果已经有至少5帧将当前关键帧视为参考帧
            if (currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
                // 就对当前关键帧的深度图进行一次填补，并计算它的平均深度
                finishCurrentKeyframe();
            else
                // 否则把该关键帧直接从keyFrameGraph中剔除。
                discardCurrentKeyframe();

            // 结束某一个关键帧的建图工作，并解开在建图时激活的activeKeyFramelock锁
            map->invalidate();
        }

        // * 启动重定位线程
        // 没启动重定位线程前isRunning为false, start()后isRunning为true
        // start relocalizer if it isnt running already
        if (!relocalizer.isRunning)
            relocalizer.start(keyFrameGraph->keyframesAll);

        // did we find a frame to relocalize with?
        // takeRelocalizeResult()会调用stop()停止重定位线程，isRunning变为false
        if (relocalizer.waitResult(50))
            takeRelocalizeResult();

        return true;
    }
}

// 初始化深度
void SlamSystem::gtDepthInit(uchar *image, float *depth, double timeStamp, int id)
{
    printf("Doing GT initialization!\n");

    currentKeyFrameMutex.lock();

    currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image));
    currentKeyFrame->setDepthFromGroundTruth(depth);

    map->initializeFromGTDepth(currentKeyFrame.get());
    keyFrameGraph->addFrame(currentKeyFrame.get());

    currentKeyFrameMutex.unlock();

    if (doSlam)
    {
        keyFrameGraph->idToKeyFrameMutex.lock();
        keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
        keyFrameGraph->idToKeyFrameMutex.unlock();
    }
    if (continuousPCOutput && outputWrapper != 0)
        outputWrapper->publishKeyframe(currentKeyFrame.get());

    printf("Done GT initialization!\n");
}

// ! 跟踪线程(主线程),在main_on_images中只在启动系统时对第一帧调用，将其设为当前关键帧，并给第一个关键帧任意初始化一个深度地图和方差
void SlamSystem::randomInit(uchar *image, double timeStamp, int id)
{
    printf("Doing Random initialization!\n");

    if (!doMapping)
        printf("WARNING: mapping is disabled, but we just initialized... THIS WILL NOT WORK! Set doMapping to true.\n");

    currentKeyFrameMutex.lock();

    // *************** 将第一帧设置为当前关键帧 ***************
    currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image));
    // 智能指针std::shared_ptr<Frame> currentKeyFrame的get()函数返回的是当前智能指针指向的对象(裸指针)，而不是当前智能指针本身。
    map->initializeRandomly(currentKeyFrame.get());
    keyFrameGraph->addFrame(currentKeyFrame.get());

    currentKeyFrameMutex.unlock();

    if (doSlam)
    {
        keyFrameGraph->idToKeyFrameMutex.lock();
        keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
        keyFrameGraph->idToKeyFrameMutex.unlock();
    }
    if (continuousPCOutput && outputWrapper != 0)
        outputWrapper->publishKeyframe(currentKeyFrame.get());

    if (displayDepthMap || depthMapScreenshotFlag)
        debugDisplayDepthMap();

    printf("Done Random initialization!\n");
}

// ! 跟踪线程(主线程)，在main_on_images中调用，而不是SlamSystem初始化时创建的子线程中调用，用于计算位姿
void SlamSystem::trackFrame(uchar *image, unsigned int frameID, bool blockUntilMapped, double timestamp)
{
    // Create new frame
    std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, image));

    // *************** 1. 更新重定位：如果跟踪出了问题就要重定位 ***************
    // Slamsystem初始化时候trackingIsGood是true，所以在第二帧之后才可能需要重定位
    if (!trackingIsGood)
    {
        // relocalizer线程的启动start()是在建图线程doMappingIteration()下，这里只是记录了一下要重定位的帧
        // relocalizer线程的启动stop()也是在建图线程doMappingIteration()下，
        relocalizer.updateCurrentFrame(trackingNewFrame);   // 记录新的要重定位的帧

        // 如果跟踪失败没有建图，建图进程会被unmappedTrackedFramesSignal条件变量卡住，这里是让建图进程继续下去
        unmappedTrackedFramesMutex.lock();  
        unmappedTrackedFramesSignal.notify_one();           // * 建图进程，每一次建图成功后都会等待条件变量unmappedTrackedFramesSignal的唤醒，至多等待200ms
        unmappedTrackedFramesMutex.unlock();
        return;
    }

    // *************** 2. 修正参考帧：如果当前正在追踪的参考帧不是当前最新的关键帧，或者当前帧的深度被更新了，就更新当前正在追踪的参考帧 ***************
    currentKeyFrameMutex.lock();
    bool my_createNewKeyframe = createNewKeyFrame; // pre-save here, to make decision afterwards.SlamSystem类初始化时为false
    /*  
        第一个判断条件：
            是判断当前正在追踪的参考帧(某一个关键帧)是不是最新的关键帧，如果不是就把当前帧导入参考帧的keyframe中去
            如果不是就把当前最新的关键帧变成参考帧
        对于第1帧：    
            SLAM系统初始化时trackingReference->keyframe = 0即空指针
            而currentKeyFrame.get()在上面的randomInit()中被设为了第一帧
            所以第一帧天生就是关键帧，是后面至少5帧的参考帧
    */
    if (trackingReference->keyframe != currentKeyFrame.get() || currentKeyFrame->depthHasBeenUpdatedFlag)
    {
        trackingReference->importFrame(currentKeyFrame.get());  
        currentKeyFrame->depthHasBeenUpdatedFlag = false;
        trackingReferenceFrameSharedPT = currentKeyFrame;
    }
    // 正在追踪的参考帧的位姿
    FramePoseStruct *trackingReferencePose = trackingReference->keyframe->pose;
    currentKeyFrameMutex.unlock();


    // *************** 3. 执行跟踪：计算位姿并显示跟踪结果 ***************
    // DO TRACKING & Show tracking result.
    if (enablePrintDebugInfo && printThreadingInfo)
        printf("TRACKING %d on %d\n", trackingNewFrame->id(), trackingReferencePose->frameID);

     // * 3.1 得到当前帧相对于参考帧的初始相对位姿
    // 这里会和回环和优化线程产生互斥，避免同时修改某一关键帧的位姿
    poseConsistencyMutex.lock_shared();
    /*
        得到当前帧相对于参考帧的初始相对位姿 = 当前帧绝对位姿的逆 * 参考帧的绝对位姿
            trackingReferencePose->getCamToWorld().inverse()： 得到当前帧的绝对位姿的逆
            
            keyFrameGraph->allFramePoses: 保存着所有帧的位姿，最后一帧即当前帧的参考帧
            back()：返回最后一个元素
    */
    SE3 frameToReference_initialEstimate = se3FromSim3(trackingReferencePose->getCamToWorld().inverse() *
                                                       keyFrameGraph->allFramePoses.back()->getCamToWorld());
    poseConsistencyMutex.unlock_shared();

    // * 3.2 计算当前帧相对于参考帧的相对位姿
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);
    // trackFrame(): 跟踪新的一帧, 主体是一个for循环，从图像金字塔的高层level-4开始遍历直到底层level-1。每一层都进行LM优化迭代，则是另外一个for循环。
    // 三个参数：1.参考帧 2.当前帧 3.当前帧相对于参考帧的初始位姿
    SE3 newRefToFrame_poseUpdate =
        tracker->trackFrame(trackingReference, trackingNewFrame.get(), frameToReference_initialEstimate);
    gettimeofday(&tv_end, NULL);

    // * 3.3 记录一些指标： 跟踪线程累计花费的时间，光度误差，像素重叠区域的比例 和 好像素的比例
    msTrackFrame = 0.9 * msTrackFrame +
                   0.1 * ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    nTrackFrame++;

    tracking_lastResidual = tracker->lastResidual;  // 平均残差（光度误差）
    tracking_lastUsage = tracker->pointUsage;       // 当前帧和参考帧重叠区域的比例
    tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);    // 如果光度误差足够小，就是一个好的像素
    tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL) *
                                                          trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));

    // *************** 4. 跟踪失败的话，需要重定位，本函数直接返回 ***************
    if (manualTrackingLossIndicated || tracker->diverged ||
        (keyFrameGraph->keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
    {
        printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
               trackingNewFrame->id(), 100 * tracking_lastGoodPerTotal, 100 * tracking_lastGoodPerBad,
               tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

        trackingReference->invalidate();

        trackingIsGood = false;
        nextRelocIdx = -1;

        // 如果跟踪失败没有建图，建图进程会被unmappedTrackedFramesSignal条件变量卡住，这里是让建图进程继续下去
        unmappedTrackedFramesMutex.lock();
        unmappedTrackedFramesSignal.notify_one();
        unmappedTrackedFramesMutex.unlock();

        manualTrackingLossIndicated = false;
        return;
    }

    // *************** 5. 绘制跟踪（debug） ***************
    if (plotTracking)
    {
        Eigen::Matrix<float, 20, 1> data;
        data.setZero();
        data[0] = tracker->lastResidual;

        data[3] = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
        data[4] = 4 * tracker->lastGoodCount / (width * height);
        data[5] = tracker->pointUsage;

        data[6] = tracker->affineEstimation_a;
        data[7] = tracker->affineEstimation_b;
        outputWrapper->publishDebugInfo(data);
    }

    // *************** 6. 跟踪成功的话将当前帧加入g2o图中 ***************
    // keyFrameGraph保存着所有成功跟踪帧的信息
    keyFrameGraph->addFrame(trackingNewFrame.get());

    // Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//
    // mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() *
    // sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
    if (outputWrapper != 0)
    {
        outputWrapper->publishTrackedFrame(trackingNewFrame.get());
    }

    // *************** 7. 关键帧选取 ***************
    latestTrackedFrame = trackingNewFrame;
    /*
        满足如下两个条件就计算得分，得分大于minVal则构建新的关键帧
        1. my_createNewKeyframe在第2步中=createNewKeyFrame，在slamsystem类初始化时是false
                            createNewKeyFrame表示当前帧是否应该被构建为关键帧，初始时每一帧都是false
        2. currentKeyFrame->numMappedOnThisTotal 根据当前关键帧(即现在的参考帧)来计算相对位姿的帧的数量不能小于MIN_NUM_MAPPED=5
                            即要求每个关键帧都必须至少要有5个普通帧的位姿是根据这个关键帧算的
    */ 
    if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
    {   
        // * 7.1 计算移动的距离
        // ? 选取关键帧根据运动距离来确定，参考论文公式16
        // newRefToFrame_poseUpdate: 刚估计得到的当前帧相对于参考帧的相对位姿
        // currentKeyFrame->meanIdepth： 参考帧的平均逆深度mean inverse depth  <------- 是公式的权重w？
        Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;

        // * 7.2 计算要求移动的距离
        // 要求相机至少运动minVal才能被创建为关键帧
        float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT, 1.0f);
        if (keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)
            minVal *= 0.7;

        // 计算运动距离: 关键帧和参考帧之间的距离平方 * 关键帧距离权重^2 * 当前帧使用的像素个数^2 * 当前帧的参考帧使用的像素个数^2
        lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);

        // * 7.3 判断是否创建为关键帧：如果当前帧相对于参考帧运动的距离 > 要求的距离，就将其创建为新的关键帧
        // 深度图的传播发生在构建关键帧的时候。在构建新的关键帧时，使用其参考关键帧的深度图来构建当前帧的深度图
        if (lastTrackingClosenessScore > minVal)
        {
            createNewKeyFrame = true;

            // debug信息
            if (enablePrintDebugInfo && printKeyframeSelectionInfo)
                printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n", trackingNewFrame->id(),
                       trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage,
                       trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
        }
        else
        {   
            // debug信息
            if (enablePrintDebugInfo && printKeyframeSelectionInfo)
                printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n", trackingNewFrame->id(),
                       trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage,
                       trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
        }
    }

    // *************** 8. unmappedTrackedFrames用来保存已经跟踪过每一帧 ***************
    // 如果没有建图，建图进程会被unmappedTrackedFramesSignal条件变量卡住，这里是让建图进程继续下去
    unmappedTrackedFramesMutex.lock();
    if (unmappedTrackedFrames.size() < 50 ||
        (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
        unmappedTrackedFrames.push_back(trackingNewFrame);
    unmappedTrackedFramesSignal.notify_one();
    unmappedTrackedFramesMutex.unlock();

    // implement blocking
    // blockUntilMapped只有在_hz被设置为0时才为true，这时候每秒读取0张新图，所以被卡住了
    if (blockUntilMapped && trackingIsGood)
    {
        boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
        while (unmappedTrackedFrames.size() > 0)
        {
            // printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
            newFrameMappedSignal.wait(lock);
        }
        lock.unlock();
    }
}

float SlamSystem::tryTrackSim3(TrackingReference *A, TrackingReference *B, int lvlStart, int lvlEnd, bool useSSE,
                               Sim3 &AtoB, Sim3 &BtoA, KFConstraintStruct *e1, KFConstraintStruct *e2)
{
    BtoA = constraintTracker->trackFrameSim3(A, B->keyframe, BtoA, lvlStart, lvlEnd);
    Matrix7x7 BtoAInfo = constraintTracker->lastSim3Hessian;
    float BtoA_meanResidual = constraintTracker->lastResidual;
    float BtoA_meanDResidual = constraintTracker->lastDepthResidual;
    float BtoA_meanPResidual = constraintTracker->lastPhotometricResidual;
    float BtoA_usage = constraintTracker->pointUsage;

    if (constraintTracker->diverged || BtoA.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
        BtoA.scale() < Sophus::SophusConstants<sophusType>::epsilon() || BtoAInfo(0, 0) == 0 || BtoAInfo(6, 6) == 0)
    {
        return 1e20;
    }

    AtoB = constraintTracker->trackFrameSim3(B, A->keyframe, AtoB, lvlStart, lvlEnd);
    Matrix7x7 AtoBInfo = constraintTracker->lastSim3Hessian;
    float AtoB_meanResidual = constraintTracker->lastResidual;
    float AtoB_meanDResidual = constraintTracker->lastDepthResidual;
    float AtoB_meanPResidual = constraintTracker->lastPhotometricResidual;
    float AtoB_usage = constraintTracker->pointUsage;

    if (constraintTracker->diverged || AtoB.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
        AtoB.scale() < Sophus::SophusConstants<sophusType>::epsilon() || AtoBInfo(0, 0) == 0 || AtoBInfo(6, 6) == 0)
    {
        return 1e20;
    }

    // Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
    Matrix7x7 datimesb_db = AtoB.cast<float>().Adj();
    Matrix7x7 diffHesse = (AtoBInfo.inverse() + datimesb_db * BtoAInfo.inverse() * datimesb_db.transpose()).inverse();
    Vector7 diff = (AtoB * BtoA).log().cast<float>();

    float reciprocalConsistency = (diffHesse * diff).dot(diff);

    if (e1 != 0 && e2 != 0)
    {
        e1->firstFrame = A->keyframe;
        e1->secondFrame = B->keyframe;
        e1->secondToFirst = BtoA;
        e1->information = BtoAInfo.cast<double>();
        e1->meanResidual = BtoA_meanResidual;
        e1->meanResidualD = BtoA_meanDResidual;
        e1->meanResidualP = BtoA_meanPResidual;
        e1->usage = BtoA_usage;

        e2->firstFrame = B->keyframe;
        e2->secondFrame = A->keyframe;
        e2->secondToFirst = AtoB;
        e2->information = AtoBInfo.cast<double>();
        e2->meanResidual = AtoB_meanResidual;
        e2->meanResidualD = AtoB_meanDResidual;
        e2->meanResidualP = AtoB_meanPResidual;
        e2->usage = AtoB_usage;

        e1->reciprocalConsistency = e2->reciprocalConsistency = reciprocalConsistency;
    }

    return reciprocalConsistency;
}

// ! 
// 候选帧要比较的关键帧在findConstraintsForNewKeyFrames()的newKFTrackingReference->importFrame(newKeyFrame);处添加为这些候选帧的参考帧了
// 参数：1.候选帧 2.约束 3. 约束 4. 候选者和关键帧初始相对位姿 5.loopclosureStrictness=1.5
void SlamSystem::testConstraint(Frame *candidate, KFConstraintStruct *&e1_out, KFConstraintStruct *&e2_out,
                                Sim3 candidateToFrame_initialEstimate, float strictness)
{
    // 将当前候选帧作为要处理的对象
    candidateTrackingReference->importFrame(candidate);

    Sim3 FtoC = candidateToFrame_initialEstimate.inverse(), CtoF = candidateToFrame_initialEstimate;
    Matrix7x7 FtoCInfo, CtoFInfo;

    float err_level3 = tryTrackSim3(newKFTrackingReference, candidateTrackingReference, // A = frame; b = candidate
                                    SIM3TRACKING_MAX_LEVEL - 1, 3, USESSE, FtoC, CtoF);

    if (err_level3 > 3000 * strictness)
    {
        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf("FAILE %d -> %d (lvl %d): errs (%.1f / - / -).", newKFTrackingReference->frameID,
                   candidateTrackingReference->frameID, 3, sqrtf(err_level3));

        e1_out = e2_out = 0;

        newKFTrackingReference->keyframe->trackingFailed.insert(
            std::pair<Frame *, Sim3>(candidate, candidateToFrame_initialEstimate));
        return;
    }

    float err_level2 = tryTrackSim3(newKFTrackingReference, candidateTrackingReference, // A = frame; b = candidate
                                    2, 2, USESSE, FtoC, CtoF);

    if (err_level2 > 4000 * strictness)
    {
        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / -).", newKFTrackingReference->frameID,
                   candidateTrackingReference->frameID, 2, sqrtf(err_level3), sqrtf(err_level2));

        e1_out = e2_out = 0;
        newKFTrackingReference->keyframe->trackingFailed.insert(
            std::pair<Frame *, Sim3>(candidate, candidateToFrame_initialEstimate));
        return;
    }

    e1_out = new KFConstraintStruct();
    e2_out = new KFConstraintStruct();

    float err_level1 = tryTrackSim3(newKFTrackingReference, candidateTrackingReference, // A = frame; b = candidate
                                    1, 1, USESSE, FtoC, CtoF, e1_out, e2_out);

    if (err_level1 > 6000 * strictness)
    {
        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / %.1f).", newKFTrackingReference->frameID,
                   candidateTrackingReference->frameID, 1, sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

        delete e1_out;
        delete e2_out;
        e1_out = e2_out = 0;
        newKFTrackingReference->keyframe->trackingFailed.insert(
            std::pair<Frame *, Sim3>(candidate, candidateToFrame_initialEstimate));
        return;
    }

    if (enablePrintDebugInfo && printConstraintSearchInfo)
        printf("ADDED %d -> %d: errs (%.1f / %.1f / %.1f).", newKFTrackingReference->frameID,
               candidateTrackingReference->frameID, sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

    const float kernelDelta = 5 * sqrt(6000 * loopclosureStrictness);
    e1_out->robustKernel = new g2o::RobustKernelHuber();
    e1_out->robustKernel->setDelta(kernelDelta);
    e2_out->robustKernel = new g2o::RobustKernelHuber();
    e2_out->robustKernel->setDelta(kernelDelta);
}

// ! 寻找可能和新关键帧达成回环的所有候选帧，然后对关键帧和所有候选帧进行双向sim3跟踪
int SlamSystem::findConstraintsForNewKeyFrames(Frame *newKeyFrame, bool forceParent, bool useFABMAP,
                                               float closeCandidatesTH)
{
    // * 如果当前关键帧没有自己的参考帧，那自然没地方去找能和自己构成回环的旧关键帧了，所以直接返回
    if (!newKeyFrame->hasTrackingParent())
    {
        newConstraintMutex.lock();
        keyFrameGraph->addKeyFrame(newKeyFrame);
        newConstraintAdded = true;
        newConstraintCreatedSignal.notify_all();
        newConstraintMutex.unlock();
        return 0;
    }
    
    // * 如果上一个关键帧和当前关键帧之间的相对位姿很小，也没必要为它再做一遍回环跟踪直接返回。
    if (!forceParent &&
        (newKeyFrame->lastConstraintTrackedCamToWorld * newKeyFrame->getScaledCamToWorld().inverse()).log().norm() < 0.01)
        return 0;
    // 记录一下当前关键帧的绝对位姿，用来给下一个关键帧做上面的对比
    newKeyFrame->lastConstraintTrackedCamToWorld = newKeyFrame->getScaledCamToWorld();

    // *************** 1. 得到所有的候选帧和他们相对于当前关键帧的初始相对位姿 ***************
    // get all potential candidates and their initial relative pose.
    std::vector<KFConstraintStruct *, Eigen::aligned_allocator<KFConstraintStruct *>> constraints;
    Frame *fabMapResult = 0;
    
    // * 在已经加入g2o的关键帧中找到，当前关键帧可以追踪到的其他关键帧，保存在无序集合unordered_set类型的candidates中
    // 注意这里useFABMAP为true时，会用fabmap回环检测算法计算是否回环，并把结果放在fabMapResult中
    std::unordered_set<Frame *, std::hash<Frame *>, std::equal_to<Frame *>, Eigen::aligned_allocator<Frame *>> candidates =
        trackableKeyFrameSearch->findCandidates(newKeyFrame, fabMapResult, useFABMAP, closeCandidatesTH);
    std::map<Frame *, Sim3, std::less<Frame *>, Eigen::aligned_allocator<std::pair<Frame *, Sim3>>>
        candidateToFrame_initialEstimateMap;

    // * 遍历所有的候选帧，如果当前关键帧和某一帧已经建立了SIM3约束，就这一帧从候选帧删除不额外算一次
    // erase the ones that are already neighbours.
    for (std::unordered_set<Frame *>::iterator c = candidates.begin(); c != candidates.end();)
    {
        if (newKeyFrame->neighbors.find(*c) != newKeyFrame->neighbors.end())
        {
            if (enablePrintDebugInfo && printConstraintSearchInfo)
                printf("SKIPPING %d on %d cause it already exists as constraint.\n", (*c)->id(), newKeyFrame->id());
            c = candidates.erase(c);
        }
        else
            ++c;
    }

    // * 得到所有候选帧相对于当前关键帧的初始相对位姿
    // 这里会和优化和跟踪线程产生互斥，避免同时修改某一关键帧的位姿
    poseConsistencyMutex.lock_shared();
    for (Frame *candidate : candidates)
    {
        Sim3 candidateToFrame_initialEstimate =
            newKeyFrame->getScaledCamToWorld().inverse() * candidate->getScaledCamToWorld();
        candidateToFrame_initialEstimateMap[candidate] = candidateToFrame_initialEstimate;
    }

    // * 如果当前关键帧有参考帧(候选帧)，计算关键帧到这些参考帧之间的距离，保存在无序列表distancesToNewKeyFrame中
    std::unordered_map<Frame *, int> distancesToNewKeyFrame;
    if (newKeyFrame->hasTrackingParent())
        keyFrameGraph->calculateGraphDistancesToFrame(newKeyFrame->getTrackingParent(), &distancesToNewKeyFrame);
    poseConsistencyMutex.unlock_shared();

    
    // distinguish between close and "far" candidates in Graph
    // Do a first check on trackability of close candidates.
    std::unordered_set<Frame *, std::hash<Frame *>, std::equal_to<Frame *>, Eigen::aligned_allocator<Frame *>>
        closeCandidates;
    std::vector<Frame *, Eigen::aligned_allocator<Frame *>> farCandidates;
    Frame *parent = newKeyFrame->hasTrackingParent() ? newKeyFrame->getTrackingParent() : 0;

    int closeFailed = 0;
    int closeInconsistent = 0;

    // 扰动：一个绕x轴旋转0.05弧度的旋转矩阵
    SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05, 0, 0));

    // *************** 2. 用双向追踪来分辨出和当前关键帧离得近的候选关键帧***************
    for (Frame *candidate : candidates)
    {
        // 如果这个候选帧就是当前关键帧，跳过
        if (candidate->id() == newKeyFrame->id())
            continue;
        // 如果这个候选帧没有在图里，说明它没有绝对位姿跳过
        if (!candidate->pose->isInGraph)
            continue;
        // 如果这个候选帧是当前关键帧的参考帧，因为已经有相对位姿了所以跳过
        if (newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
            continue;
        // 如果候选帧队列里的关键帧<5了跳过
        // idxInKeyframes在Frame被初始化时是-1，INITIALIZATION_PHASE_COUNT=5
        if (candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
            continue;

        // * 2.1 论文中提到的reciprocal tracking check相互追踪检查
        // ** 计算当前关键帧frame与其跟踪到的候选帧candidate之间的相对位姿
        // 将候选帧与当前关键帧的相对位姿从sim3->se3
        SE3 c2f_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate].inverse()).inverse();
        // 给这个初始相对位姿的旋转矩阵加一个小扰动
        c2f_init.so3() = c2f_init.so3() * disturbance;
        // 计算当前关键帧frame与其跟踪到的候选帧candidate之间的相对位姿
        SE3 c2f = constraintSE3Tracker->trackFrameOnPermaref(candidate, newKeyFrame, c2f_init);
        if (!constraintSE3Tracker->trackingWasGood)
        {
            closeFailed++;
            continue;
        }

        // ** 计算当前关键帧跟踪到的候选帧candidate与自己frame之间的相对位姿
        SE3 f2c_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate]).inverse();
        f2c_init.so3() = disturbance * f2c_init.so3();
        SE3 f2c = constraintSE3Tracker->trackFrameOnPermaref(newKeyFrame, candidate, f2c_init);
        if (!constraintSE3Tracker->trackingWasGood)
        {
            closeFailed++;
            continue;
        }

        // ** 比较双向追踪的结果，如果两者差异太大，这个候选帧和关键帧就不构成回环
        if ((f2c.so3() * c2f.so3()).log().norm() >= 0.09)
        {
            closeInconsistent++;
            continue;
        }

        // * 2.2 挺过了双向追踪，这个候选帧离被判定为和当前关键帧构成回环不远了
        closeCandidates.insert(candidate);
    }

    // *************** 3. 使用fabmap回环检测算法来得到和当前关键帧离得远的候选关键帧 ***************.
    int farFailed = 0;
    int farInconsistent = 0;
    for (Frame *candidate : candidates)
    {
        if (candidate->id() == newKeyFrame->id())
            continue;
        if (!candidate->pose->isInGraph)
            continue;
        if (newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
            continue;
        // idxInKeyframes在Frame被初始化时是-1
        if (candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
            continue;

        // * 使用fabmap回环检测的结果，如果是就加入farCandidates
        if (candidate == fabMapResult)
        {
            farCandidates.push_back(candidate);
            continue;
        }

        // 如果关键帧与该候选帧之间的距离<4直接返回
        if (distancesToNewKeyFrame.at(candidate) < 4)
            continue;

        // * 如果上面这些直接返回条件都不满足，也将其加入farCandidates
        farCandidates.push_back(candidate);
    }

    int closeAll = closeCandidates.size();
    int farAll = farCandidates.size();

    // *************** 4. 删除已经尝试过的离得近的候选关键帧 ***************
    // erase the ones that we tried already before (close)
    for (std::unordered_set<Frame *>::iterator c = closeCandidates.begin(); c != closeCandidates.end();)
    {
        // 如果这一参考帧已经在失败表了，直接返回
        if (newKeyFrame->trackingFailed.find(*c) == newKeyFrame->trackingFailed.end())
        {
            ++c;
            continue;
        }
        // trackingFailed是一个哈希表std::unordered_multimap类型，equal_range()返回所有键值==c的元素
        // 这个哈希表的键是：Frame*，值是：Sim3
        auto range = newKeyFrame->trackingFailed.equal_range(*c);

        bool skip = false;

        // * 如果初始相对位姿 与 双向跟踪过的相对位姿 之间的差异<0.1，说明这个候选帧已经被相同手法处理过了
        Sim3 f2c = candidateToFrame_initialEstimateMap[*c].inverse();
        for (auto it = range.first; it != range.second; ++it)
        {
            if ((f2c * it->second).log().norm() < 0.1)
            {
                skip = true;
                break;
            }
        }

        // * 删除这个已经处理过的候选帧
        if (skip)
        {
            if (enablePrintDebugInfo && printConstraintSearchInfo)
                printf("SKIPPING %d on %d (NEAR), cause we already have tried it.\n", (*c)->id(), newKeyFrame->id());
            c = closeCandidates.erase(c);
        }
        else
            ++c;
    }
    
    // *************** 4. 删除已经尝试过的离得远的候选关键帧 ***************
    // erase the ones that are already neighbours (far)
    for (unsigned int i = 0; i < farCandidates.size(); i++)
    {
        if (newKeyFrame->trackingFailed.find(farCandidates[i]) == newKeyFrame->trackingFailed.end())
            continue;

        auto range = newKeyFrame->trackingFailed.equal_range(farCandidates[i]);

        bool skip = false;

        // * 如果初始相对位姿 与 fabmap跟踪过的相对位姿 之间的差异<0.1，说明这个候选帧已经被相同手法处理过了
        for (auto it = range.first; it != range.second; ++it)
        {
            if ((it->second).log().norm() < 0.2)
            {
                skip = true;
                break;
            }
        }

        // * 删除这个已经处理过的候选帧
        if (skip)
        {
            if (enablePrintDebugInfo && printConstraintSearchInfo)
                printf("SKIPPING %d on %d (FAR), cause we already have tried it.\n", farCandidates[i]->id(), newKeyFrame->id());
            farCandidates[i] = farCandidates.back();
            farCandidates.pop_back();
            i--;
        }
    }

    if (enablePrintDebugInfo && printConstraintSearchInfo)
        printf("Final Loop-Closure Candidates: %d / %d close (%d failed, %d inconsistent) + %d / %d far (%d failed, %d "
               "inconsistent) = %d\n",
               (int)closeCandidates.size(), closeAll, closeFailed, closeInconsistent, (int)farCandidates.size(), farAll,
               farFailed, farInconsistent, (int)closeCandidates.size() + (int)farCandidates.size());

    // *************** 5. 限制离得近的候选关键帧个数<10 ***************
    // while too many, remove the one with the highest connectivity.
    // setting.cpp中maxLoopClosureCandidates = 10
    while ((int)closeCandidates.size() > maxLoopClosureCandidates)
    {
        Frame *worst = 0;
        int worstNeighbours = 0;
        for (Frame *f : closeCandidates)
        {
            int neightboursInCandidates = 0;
            for (Frame *n : f->neighbors)
                if (closeCandidates.find(n) != closeCandidates.end())
                    neightboursInCandidates++;

            if (neightboursInCandidates > worstNeighbours || worst == 0)
            {
                worst = f;
                worstNeighbours = neightboursInCandidates;
            }
        }

        closeCandidates.erase(worst);
    }

    // *************** 6. 限制离得远的候选关键帧个数<5 ***************
    // delete randomly
    int maxNumFarCandidates = (maxLoopClosureCandidates + 1) / 2;
    if (maxNumFarCandidates < 5)
        maxNumFarCandidates = 5;
    while ((int)farCandidates.size() > maxNumFarCandidates)
    {
        int toDelete = rand() % farCandidates.size();
        if (farCandidates[toDelete] != fabMapResult)
        {
            farCandidates[toDelete] = farCandidates.back();
            farCandidates.pop_back();
        }
    }

    // =============== TRACK! ===============

    // 将当前关键帧作为这些候选帧的参考帧
    // make tracking reference for newKeyFrame.
    newKFTrackingReference->importFrame(newKeyFrame);

    for (Frame *candidate : closeCandidates)
    {
        KFConstraintStruct *e1 = 0; // 如果某两帧构成元素保存在此结构
        KFConstraintStruct *e2 = 0;

        // 参数：1.候选帧 2.约束 3. 约束 4. 候选者和关键帧初始相对位姿 5.loopclosureStrictness=1.5
        testConstraint(candidate, e1, e2, candidateToFrame_initialEstimateMap[candidate], loopclosureStrictness);

        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf(" CLOSE (%d)\n", distancesToNewKeyFrame.at(candidate));

        if (e1 != 0)
        {
            constraints.push_back(e1);
            constraints.push_back(e2);

            // delete from far candidates if it's in there.
            for (unsigned int k = 0; k < farCandidates.size(); k++)
            {
                if (farCandidates[k] == candidate)
                {
                    if (enablePrintDebugInfo && printConstraintSearchInfo)
                        printf(" DELETED %d from far, as close was successful!\n", candidate->id());

                    farCandidates[k] = farCandidates.back();
                    farCandidates.pop_back();
                }
            }
        }
    }

    for (Frame *candidate : farCandidates)
    {
        KFConstraintStruct *e1 = 0;
        KFConstraintStruct *e2 = 0;

        testConstraint(candidate, e1, e2, Sim3(), loopclosureStrictness);

        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf(" FAR (%d)\n", distancesToNewKeyFrame.at(candidate));

        if (e1 != 0)
        {
            constraints.push_back(e1);
            constraints.push_back(e2);
        }
    }

    if (parent != 0 && forceParent)
    {
        KFConstraintStruct *e1 = 0;
        KFConstraintStruct *e2 = 0;
        testConstraint(parent, e1, e2, candidateToFrame_initialEstimateMap[parent], 100);
        if (enablePrintDebugInfo && printConstraintSearchInfo)
            printf(" PARENT (0)\n");

        if (e1 != 0)
        {
            constraints.push_back(e1);
            constraints.push_back(e2);
        }
        else
        {
            float downweightFac = 5;
            const float kernelDelta = 5 * sqrt(6000 * loopclosureStrictness) / downweightFac;
            printf("warning: reciprocal tracking on new frame failed badly, added odometry edge (Hacky).\n");

            poseConsistencyMutex.lock_shared();
            constraints.push_back(new KFConstraintStruct());
            constraints.back()->firstFrame = newKeyFrame;
            constraints.back()->secondFrame = newKeyFrame->getTrackingParent();
            constraints.back()->secondToFirst = constraints.back()->firstFrame->getScaledCamToWorld().inverse() *
                                                constraints.back()->secondFrame->getScaledCamToWorld();
            constraints.back()->information << 0.8098, -0.1507, -0.0557, 0.1211, 0.7657, 0.0120, 0, -0.1507, 2.1724, -0.1103,
                -1.9279, -0.1182, 0.1943, 0, -0.0557, -0.1103, 0.2643, -0.0021, -0.0657, -0.0028, 0.0304, 0.1211, -1.9279,
                -0.0021, 2.3110, 0.1039, -0.0934, 0.0005, 0.7657, -0.1182, -0.0657, 0.1039, 1.0545, 0.0743, -0.0028, 0.0120,
                0.1943, -0.0028, -0.0934, 0.0743, 0.4511, 0, 0, 0, 0.0304, 0.0005, -0.0028, 0, 0.0228;
            constraints.back()->information *= (1e9 / (downweightFac * downweightFac));

            constraints.back()->robustKernel = new g2o::RobustKernelHuber();
            constraints.back()->robustKernel->setDelta(kernelDelta);

            constraints.back()->meanResidual = 10;
            constraints.back()->meanResidualD = 10;
            constraints.back()->meanResidualP = 10;
            constraints.back()->usage = 0;

            poseConsistencyMutex.unlock_shared();
        }
    }

    newConstraintMutex.lock();

    keyFrameGraph->addKeyFrame(newKeyFrame);
    for (unsigned int i = 0; i < constraints.size(); i++)
        keyFrameGraph->insertConstraint(constraints[i]);

    newConstraintAdded = true;
    newConstraintCreatedSignal.notify_all();
    newConstraintMutex.unlock();

    newKFTrackingReference->invalidate();
    candidateTrackingReference->invalidate();

    return constraints.size();
}

bool SlamSystem::optimizationIteration(int itsPerTry, float minChange)
{
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    g2oGraphAccessMutex.lock();

    // lock new elements buffer & take them over.
    newConstraintMutex.lock();
    keyFrameGraph->addElementsFromBuffer();
    newConstraintMutex.unlock();

    // Do the optimization. This can take quite some time!
    int its = keyFrameGraph->optimize(itsPerTry);

    // save the optimization result.
    poseConsistencyMutex.lock_shared();
    keyFrameGraph->keyframesAllMutex.lock_shared();
    float maxChange = 0;
    float sumChange = 0;
    float sum = 0;
    for (size_t i = 0; i < keyFrameGraph->keyframesAll.size(); i++)
    {
        // set edge error sum to zero
        keyFrameGraph->keyframesAll[i]->edgeErrorSum = 0;
        keyFrameGraph->keyframesAll[i]->edgesNum = 0;

        if (!keyFrameGraph->keyframesAll[i]->pose->isInGraph)
            continue;

        // get change from last optimization
        Sim3 a = keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate();
        Sim3 b = keyFrameGraph->keyframesAll[i]->getScaledCamToWorld();
        Sophus::Vector7f diff = (a * b.inverse()).log().cast<float>();

        for (int j = 0; j < 7; j++)
        {
            float d = fabsf((float)(diff[j]));
            if (d > maxChange)
                maxChange = d;
            sumChange += d;
        }
        sum += 7;

        // set change
        keyFrameGraph->keyframesAll[i]->pose->setPoseGraphOptResult(
            keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate());

        // add error
        for (auto edge : keyFrameGraph->keyframesAll[i]->pose->graphVertex->edges())
        {
            keyFrameGraph->keyframesAll[i]->edgeErrorSum += ((EdgeSim3 *)(edge))->chi2();
            keyFrameGraph->keyframesAll[i]->edgesNum++;
        }
    }

    haveUnmergedOptimizationOffset = true;
    keyFrameGraph->keyframesAllMutex.unlock_shared();
    poseConsistencyMutex.unlock_shared();

    g2oGraphAccessMutex.unlock();

    if (enablePrintDebugInfo && printOptimizationInfo)
        printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", its, maxChange,
               sumChange / sum,
               maxChange > minChange && its == itsPerTry ? "continue optimizing" : "Waiting for addition to graph.");

    gettimeofday(&tv_end, NULL);
    msOptimizationIteration = 0.9 * msOptimizationIteration + 0.1 * ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                                     (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    nOptimizationIteration++;

    return maxChange > minChange && its == itsPerTry;
}

void SlamSystem::optimizeGraph()
{
    boost::unique_lock<boost::mutex> g2oLock(g2oGraphAccessMutex);
    keyFrameGraph->optimize(1000);
    g2oLock.unlock();
    mergeOptimizationOffset();
}

SE3 SlamSystem::getCurrentPoseEstimate()
{
    SE3 camToWorld = SE3();
    keyFrameGraph->allFramePosesMutex.lock_shared();
    if (keyFrameGraph->allFramePoses.size() > 0)
        camToWorld = se3FromSim3(keyFrameGraph->allFramePoses.back()->getCamToWorld());
    keyFrameGraph->allFramePosesMutex.unlock_shared();
    return camToWorld;
}

std::vector<FramePoseStruct *, Eigen::aligned_allocator<FramePoseStruct *>> SlamSystem::getAllPoses()
{
    return keyFrameGraph->allFramePoses;
}
