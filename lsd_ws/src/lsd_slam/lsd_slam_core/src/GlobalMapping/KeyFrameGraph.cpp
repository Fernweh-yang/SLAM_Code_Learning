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

#include "KeyFrameGraph.h"
#include "DataStructures/Frame.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalFuncs.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/types/sim3/sim3.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include "opencv2/opencv.hpp"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>
// for iterating over files in a directory
#include <dirent.h>
#include <queue>

#include <memory>
#include <iostream>
#include <fstream>

namespace lsd_slam
{
    KFConstraintStruct::~KFConstraintStruct()
    {
        if (edge != 0)
            delete edge;
    }

    // 构造函数，构建一个新的g2o图优化
    KeyFrameGraph::KeyFrameGraph() : nextEdgeId(0)
    {   
        // BlockSolver 是一个通用的求解器框架，用于求解大多数图优化问题
        // 指定块矩阵的维度，每个误差项优化变量维度为7，误差值维度为3
        typedef g2o::BlockSolver_7_3 BlockSolver;
        
        // LinearSolverCSparse是g2o 中使用 CSparse 库实现的线性求解器， CSparse 是一个轻量级的稀疏矩阵操作库，用于处理稀疏矩阵
        // PoseMatrixType 是 BlockSolver 类的一个类型别名，表示用于存储位姿块的稠密矩阵的类型
        typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;

        // create the linear solver
        // 创建上面设置好的线性求解器
        auto solver = g2o::make_unique<LinearSolver>();
        // create the block solver on top of the linear solver
        // 创建上面设置好的块求解器
        auto blockSolver = g2o::make_unique<BlockSolver>(std::move(solver));
        // create the algorithm to carry out the optimization
        // 使用L-M算法来做优化
        auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
        graph.setAlgorithm(algorithm);

        // printOptimizationInfo
        // 关闭/开启优化中输出的状态信息
        graph.setVerbose(false);

        // the following lines cause segfaults
        //  solver->setWriteDebug(true);
        //  blockSolver->setWriteDebug(true);
        //  algorithm->setWriteDebug(true);

        totalPoints = 0;
        totalEdges = 0;
        totalVertices = 0;
    }

    KeyFrameGraph::~KeyFrameGraph()
    {
        // deletes edges
        for (KFConstraintStruct *edge : newEdgeBuffer)
            delete edge; // deletes the g2oedge, which deletes the kernel.

        // deletes keyframes (by deleting respective shared pointers).

        idToKeyFrame.clear();

        // deletes pose structs (their memory is managed by graph)
        // WARNING: at this point, all Frames have to be deleted, otherwise it night cause segfaults!
        for (FramePoseStruct *p : allFramePoses)
            delete p;
    }

    // ! 将某一帧保存到图优化g2o数据结构中去，但并不是添加为顶点，下面的addKeyFrame()函数会添加为顶点
    void KeyFrameGraph::addFrame(Frame *frame)
    {
        frame->pose->isRegisteredToGraph = true;
        FramePoseStruct *pose = frame->pose;

        allFramePosesMutex.lock();
        allFramePoses.push_back(pose);
        allFramePosesMutex.unlock();
    }

    // ! 将关键帧图优化的结果以及相关的信息保存到指定的文件夹中
    void KeyFrameGraph::dumpMap(std::string folder)
    {
        printf("DUMP MAP: dumping to %s\n", folder.c_str());

        keyframesAllMutex.lock_shared();
        char buf[100];
        int succ = system(("rm -rf " + folder).c_str());
        succ += system(("mkdir " + folder).c_str());

        // 遍历所有的关键帧（keyframesAll），并将它们的深度图、灰度图以及深度方差图保存为相应的图像文件。
        for (unsigned int i = 0; i < keyframesAll.size(); i++)
        {
            snprintf(buf, 100, "%s/depth-%d.png", folder.c_str(), i);
            cv::imwrite(buf, getDepthRainbowPlot(keyframesAll[i], 0));

            snprintf(buf, 100, "%s/frame-%d.png", folder.c_str(), i);
            cv::imwrite(buf, cv::Mat(keyframesAll[i]->height(), keyframesAll[i]->width(), CV_32F, keyframesAll[i]->image()));

            snprintf(buf, 100, "%s/var-%d.png", folder.c_str(), i);
            cv::imwrite(buf, getVarRedGreenPlot(keyframesAll[i]->idepthVar(), keyframesAll[i]->image(),
                                                keyframesAll[i]->width(), keyframesAll[i]->height()));
        }

        // 显示深度方差图
        int i = keyframesAll.size() - 1;
        Util::displayImage("VAR PREVIEW", getVarRedGreenPlot(keyframesAll[i]->idepthVar(), keyframesAll[i]->image(),
                                                             keyframesAll[i]->width(), keyframesAll[i]->height()));

        printf("DUMP MAP (succ %d): dumped %d depthmaps\n", succ, (int)keyframesAll.size());

        Eigen::MatrixXf res, resD, resP, huber, usage, consistency, distance, error;
        Eigen::VectorXf meanRootInformation, usedPixels;

        res.resize(keyframesAll.size(), keyframesAll.size());
        resD.resize(keyframesAll.size(), keyframesAll.size());
        resP.resize(keyframesAll.size(), keyframesAll.size());
        usage.resize(keyframesAll.size(), keyframesAll.size());
        consistency.resize(keyframesAll.size(), keyframesAll.size());
        distance.resize(keyframesAll.size(), keyframesAll.size());
        error.resize(keyframesAll.size(), keyframesAll.size());
        meanRootInformation.resize(keyframesAll.size());
        usedPixels.resize(keyframesAll.size());
        res.setZero();
        resD.setZero();
        resP.setZero();
        usage.setZero();
        consistency.setZero();
        distance.setZero();
        error.setZero();
        meanRootInformation.setZero();
        usedPixels.setZero();

        for (unsigned int i = 0; i < keyframesAll.size(); i++)
        {
            meanRootInformation[i] = keyframesAll[i]->meanInformation;
            usedPixels[i] = keyframesAll[i]->numPoints / (float)keyframesAll[i]->numMappablePixels;
        }

        edgesListsMutex.lock_shared();
        for (unsigned int i = 0; i < edgesAll.size(); i++)
        {
            KFConstraintStruct *e = edgesAll[i];

            res(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidual;
            resD(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidualD;
            resP(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidualP;
            usage(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->usage;
            consistency(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->reciprocalConsistency;
            distance(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->secondToFirst.translation().norm();
            error(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->edge->chi2();
        }
        edgesListsMutex.unlock_shared();
        keyframesAllMutex.unlock_shared();

        // 保存信息到文本文件
        std::ofstream fle;

        fle.open(folder + "/residual.txt");
        fle << res;
        fle.close();

        fle.open(folder + "/residualD.txt");
        fle << resD;
        fle.close();

        fle.open(folder + "/residualP.txt");
        fle << resP;
        fle.close();

        fle.open(folder + "/usage.txt");
        fle << usage;
        fle.close();

        fle.open(folder + "/consistency.txt");
        fle << consistency;
        fle.close();

        fle.open(folder + "/distance.txt");
        fle << distance;
        fle.close();

        fle.open(folder + "/error.txt");
        fle << error;
        fle.close();

        fle.open(folder + "/meanRootInformation.txt");
        fle << meanRootInformation;
        fle.close();

        fle.open(folder + "/usedPixels.txt");
        fle << usedPixels;
        fle.close();

        printf("DUMP MAP: dumped %d edges\n", (int)edgesAll.size());
    }

    // ! 将新的关键帧添加到位姿图中去，作为一个顶点
    void KeyFrameGraph::addKeyFrame(Frame *frame)
    {   
        // 如果当前帧已经被加入位姿图了就返回
        if (frame->pose->graphVertex != nullptr)
            return;

        // Insert vertex into g2o graph
        VertexSim3 *vertex = new VertexSim3();
        vertex->setId(frame->id());

        Sophus::Sim3d camToWorld_estimate = frame->getScaledCamToWorld();

        // 如果当前frame没有正在跟踪的父帧(针对初始帧和关键帧），就将当前帧设置为固定，在优化时不在变动
        if (!frame->hasTrackingParent())
            vertex->setFixed(true);

        // 设置顶点的初始值(前端算到的帧的pose)
        vertex->setEstimate(camToWorld_estimate);
        // 设置顶点是否被边缘化(降低维度)
        vertex->setMarginalized(false);

        frame->pose->graphVertex = vertex;

        newKeyframesBuffer.push_back(frame);
    }

    // ! 向关键帧图中添加一个新的约束（constraint），这个约束用 EdgeSim3 类表示
    void KeyFrameGraph::insertConstraint(KFConstraintStruct *constraint)
    {   
        // 图优化中的一个相似性变换边，用于保存这个新的约束，
        EdgeSim3 *edge = new EdgeSim3();
        edge->setId(nextEdgeId);
        ++nextEdgeId;

        totalEdges++;

        // 设置边的值，即相似性变换矩阵Sim3
        edge->setMeasurement(constraint->secondToFirst);
        // 设置边的信息矩阵，用于权衡测量的重要性
        edge->setInformation(constraint->information);
        // 设置边的鲁棒核函数，用于鲁棒优化。
        edge->setRobustKernel(constraint->robustKernel);

        // 边有2个顶点（图优化），所以设置为2，并设置2个顶点（关键帧的位姿）
        edge->resize(2);
        assert(constraint->firstFrame->pose->graphVertex != nullptr);
        edge->setVertex(0, constraint->firstFrame->pose->graphVertex);
        assert(constraint->secondFrame->pose->graphVertex != nullptr);
        edge->setVertex(1, constraint->secondFrame->pose->graphVertex);

        constraint->edge = edge;
        newEdgeBuffer.push_back(constraint);

        constraint->firstFrame->neighbors.insert(constraint->secondFrame);
        constraint->secondFrame->neighbors.insert(constraint->firstFrame);

        for (int i = 0; i < totalVertices; i++)
        {
            // shortestDistancesMap
        }

        edgesListsMutex.lock();
        constraint->idxInAllEdges = edgesAll.size();
        // 将约束添加到总的边列表 edgesAll 中。
        edgesAll.push_back(constraint);
        edgesListsMutex.unlock();
    }

    // ! 将缓冲区中的新关键帧和新约束添加到图优化问题中。
    bool KeyFrameGraph::addElementsFromBuffer()
    {
        bool added = false;

        keyframesForRetrackMutex.lock();
        // 遍历每个新增的关键帧
        for (auto newKF : newKeyframesBuffer)
        {   
            // 将心关键帧的位姿顶点加入到图中
            graph.addVertex(newKF->pose->graphVertex);
            assert(!newKF->pose->isInGraph);
            newKF->pose->isInGraph = true;
            // 将新的关键帧保存到keyframesForRetrack队列中，在一致性约束中会检测这个双端队列有多少关键帧了
            keyframesForRetrack.push_back(newKF);

            added = true;
        }
        keyframesForRetrackMutex.unlock();

        newKeyframesBuffer.clear();
        // 遍历所有新的约束
        for (auto edge : newEdgeBuffer)
        {   
            // 将新约束添加到图中
            graph.addEdge(edge->edge);
            added = true;
        }
        newEdgeBuffer.clear();

        return added;
    }

    // ! 优化g2o
    int KeyFrameGraph::optimize(int num_iterations)
    {
        // Abort if graph is empty, g2o shows an error otherwise
        // 检查图是否为空
        if (graph.edges().size() == 0)
            return 0;

        // 不输出优化时的各种信息
        graph.setVerbose(false); // printOptimizationInfo
        // 初始化图优化
        graph.initializeOptimization();
        // 调用g2o的optimize()来执行图优化，迭代num_iterations次，false表示优化过程中禁用输出详细信息
        // 返回的是迭代的次数，如果一切正常，返回值==num_iterations
        return graph.optimize(num_iterations, false);
    }

    // ! 计算一个给定的帧到图中所有其他帧的最短距离，并保存在distanceMap中
    void KeyFrameGraph::calculateGraphDistancesToFrame(Frame *startFrame, std::unordered_map<Frame *, int> *distanceMap)
    {   
        // 将给定的帧插入distanceMap
        distanceMap->insert(std::make_pair(startFrame, 0));
        // 使用优先队列进行最短路径计算
        std::multimap<int, Frame *> priorityQueue;
        // 将给定的帧插入priorityQueue
        priorityQueue.insert(std::make_pair(0, startFrame));
        
        // 遍历优先队列
        while (!priorityQueue.empty())
        {
            auto it = priorityQueue.begin();
            int length = it->first;
            Frame *frame = it->second;
            priorityQueue.erase(it);    // 删除当前元素

            auto mapEntry = distanceMap->find(frame);
            // 如果 distanceMap 中已经存在 frame 的条目，且已知的距离小于等于当前计算的距离，
            // 则忽略当前帧并进入下一个循环，因为已经存在一个更短的路径。
            if (mapEntry != distanceMap->end() && length > mapEntry->second)
            {
                continue;
            }

            // 遍历给定frame的所有邻居帧 neighbor，找到最小的距离
            for (Frame *neighbor : frame->neighbors)
            {
                auto neighborMapEntry = distanceMap->find(neighbor);

                if (neighborMapEntry != distanceMap->end() && length + 1 >= neighborMapEntry->second)
                    continue;

                if (neighborMapEntry != distanceMap->end())
                    neighborMapEntry->second = length + 1;
                else
                    distanceMap->insert(std::make_pair(neighbor, length + 1));
                priorityQueue.insert(std::make_pair(length + 1, neighbor));
            }
        }
    }

} // namespace lsd_slam
