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

#include "SE3Tracker.h"
#include <opencv2/highgui/highgui.hpp>
#include "DataStructures/Frame.h"
#include "Tracking/TrackingReference.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "Tracking/least_squares.h"

#include <Eigen/Core>

namespace lsd_slam
{
#if defined(ENABLE_NEON)
#define callOptimized(function, arguments) function##NEON arguments
#else
#if defined(ENABLE_SSE)
#define callOptimized(function, arguments) (USESSE ? function##SSE arguments : function arguments)
#else
#define callOptimized(function, arguments) function arguments
#endif
#endif

    // 构造函数
    SE3Tracker::SE3Tracker(int w, int h, Eigen::Matrix3f K)
    {   
        // 相机参数
        width = w;
        height = h;
        this->K = K;
        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);

        // tracking得到配置参数
        settings = DenseDepthTrackerSettings();
        // settings.maxItsPerLvl[0] = 2;

        KInv = K.inverse();
        fxi = KInv(0, 0);
        fyi = KInv(1, 1);
        cxi = KInv(0, 2);
        cyi = KInv(1, 2);

        // 为各个需要矫正的数据分配内存
        buf_warped_residual = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_warped_dx = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_warped_dy = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_warped_x = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_warped_y = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_warped_z = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));

        buf_d = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_idepthVar = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        buf_weight_p = (float *)Eigen::internal::aligned_malloc(w * h * sizeof(float));

        buf_warped_size = 0;

        debugImageWeights = cv::Mat(height, width, CV_8UC3);
        debugImageResiduals = cv::Mat(height, width, CV_8UC3);
        debugImageSecondFrame = cv::Mat(height, width, CV_8UC3);
        debugImageOldImageWarped = cv::Mat(height, width, CV_8UC3);
        debugImageOldImageSource = cv::Mat(height, width, CV_8UC3);

        lastResidual = 0;
        iterationNumber = 0;
        pointUsage = 0;
        lastGoodCount = lastBadCount = 0;

        diverged = false;
    }

    SE3Tracker::~SE3Tracker()
    {
        debugImageResiduals.release();
        debugImageWeights.release();
        debugImageSecondFrame.release();
        debugImageOldImageSource.release();
        debugImageOldImageWarped.release();

        Eigen::internal::aligned_free((void *)buf_warped_residual);
        Eigen::internal::aligned_free((void *)buf_warped_dx);
        Eigen::internal::aligned_free((void *)buf_warped_dy);
        Eigen::internal::aligned_free((void *)buf_warped_x);
        Eigen::internal::aligned_free((void *)buf_warped_y);
        Eigen::internal::aligned_free((void *)buf_warped_z);

        Eigen::internal::aligned_free((void *)buf_d);
        Eigen::internal::aligned_free((void *)buf_idepthVar);
        Eigen::internal::aligned_free((void *)buf_weight_p);
    }

    // tracks a frame.
    // first_frame has depth, second_frame DOES NOT have depth.
    // ! 返回当前帧与参考帧之间的参考点的重叠度，可以理解为当前帧和参考帧重叠区域的比例
    float SE3Tracker::checkPermaRefOverlap(Frame *reference, SE3 referenceToFrameOrg)
    {
        Sophus::SE3f referenceToFrame = referenceToFrameOrg.cast<float>();
        boost::unique_lock<boost::mutex> lock2 = boost::unique_lock<boost::mutex>(reference->permaRef_mutex);

        // ***************** 获取参考帧的参数 ***************** 
        int w2 = reference->width(QUICK_KF_CHECK_LVL) - 1;
        int h2 = reference->height(QUICK_KF_CHECK_LVL) - 1;
        Eigen::Matrix3f KLvl = reference->K(QUICK_KF_CHECK_LVL);
        float fx_l = KLvl(0, 0);
        float fy_l = KLvl(1, 1);
        float cx_l = KLvl(0, 2);
        float cy_l = KLvl(1, 2);

        // ***************** 获取参考帧到当前帧的变换矩阵：R和t ***************** 
        Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
        Eigen::Vector3f transVec = referenceToFrame.translation();

        // ***************** 遍历所有永久(permanent?)参考点 ***************** 
        const Eigen::Vector3f *refPoint_max = reference->permaRef_posData + reference->permaRefNumPts;
        const Eigen::Vector3f *refPoint = reference->permaRef_posData;
        float usageCount = 0;
        for (; refPoint < refPoint_max; refPoint++)
        {
            // 将参考点坐标从参考帧转到当前正在追踪的帧
            Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
            float u_new = (Wxp[0] / Wxp[2]) * fx_l + cx_l;
            float v_new = (Wxp[1] / Wxp[2]) * fy_l + cy_l;
            // 确保像素坐标在当前帧图像的有效范围内
            if ((u_new > 0 && v_new > 0 && u_new < w2 && v_new < h2))
            {
                // 计算深度变化，并累加到usageCount中
                // 如果深度(*refPoint)[2] < Z Wxp[2]: 那么+一个[0,1]的小数，否则加1
                float depthChange = (*refPoint)[2] / Wxp[2];
                usageCount += depthChange < 1 ? depthChange : 1;
            }
        }
        // ***************** 计算重叠度：深度变化的累积值/永久参考点的总数 ***************** 
        pointUsage = usageCount / (float)reference->permaRefNumPts;
        return pointUsage;
    }

    // tracks a frame.
    // first_frame has depth, second_frame DOES NOT have depth.
    SE3 SE3Tracker::trackFrameOnPermaref(Frame *reference, Frame *frame, SE3 referenceToFrameOrg)
    {
        Sophus::SE3f referenceToFrame = referenceToFrameOrg.cast<float>();

        boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
        boost::unique_lock<boost::mutex> lock2 = boost::unique_lock<boost::mutex>(reference->permaRef_mutex);

        affineEstimation_a = 1;
        affineEstimation_b = 0;

        NormalEquationsLeastSquares ls;
        diverged = false;
        trackingWasGood = true;

        callOptimized(calcResidualAndBuffers,
                      (reference->permaRef_posData, reference->permaRef_colorAndVarData, 0, reference->permaRefNumPts, frame,
                       referenceToFrame, QUICK_KF_CHECK_LVL, false));
        if (buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width >> QUICK_KF_CHECK_LVL) * (height >> QUICK_KF_CHECK_LVL))
        {
            diverged = true;
            trackingWasGood = false;
            return SE3();
        }
        if (useAffineLightningEstimation)
        {
            affineEstimation_a = affineEstimation_a_lastIt;
            affineEstimation_b = affineEstimation_b_lastIt;
        }
        float lastErr = callOptimized(calcWeightsAndResidual, (referenceToFrame));

        float LM_lambda = settings.lambdaInitialTestTrack;

        for (int iteration = 0; iteration < settings.maxItsTestTrack; iteration++)
        {
            callOptimized(calculateWarpUpdate, (ls));

            int incTry = 0;
            while (true)
            {
                // solve LS system with current lambda
                Vector6 b = -ls.b;
                Matrix6x6 A = ls.A;
                for (int i = 0; i < 6; i++)
                    A(i, i) *= 1 + LM_lambda;
                Vector6 inc = A.ldlt().solve(b);
                incTry++;

                // apply increment. pretty sure this way round is correct, but hard to test.
                Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;

                // re-evaluate residual
                callOptimized(calcResidualAndBuffers,
                              (reference->permaRef_posData, reference->permaRef_colorAndVarData, 0, reference->permaRefNumPts,
                               frame, new_referenceToFrame, QUICK_KF_CHECK_LVL, false));
                if (buf_warped_size <
                    MIN_GOODPERALL_PIXEL_ABSMIN * (width >> QUICK_KF_CHECK_LVL) * (height >> QUICK_KF_CHECK_LVL))
                {
                    diverged = true;
                    trackingWasGood = false;
                    return SE3();
                }
                float error = callOptimized(calcWeightsAndResidual, (new_referenceToFrame));

                // accept inc?
                if (error < lastErr)
                {
                    // accept inc
                    referenceToFrame = new_referenceToFrame;
                    if (useAffineLightningEstimation)
                    {
                        affineEstimation_a = affineEstimation_a_lastIt;
                        affineEstimation_b = affineEstimation_b_lastIt;
                    }
                    // converged?
                    if (error / lastErr > settings.convergenceEpsTestTrack)
                        iteration = settings.maxItsTestTrack;

                    lastErr = error;

                    if (LM_lambda <= 0.2)
                        LM_lambda = 0;
                    else
                        LM_lambda *= settings.lambdaSuccessFac;

                    break;
                }
                else
                {
                    if (!(inc.dot(inc) > settings.stepSizeMinTestTrack))
                    {
                        iteration = settings.maxItsTestTrack;
                        break;
                    }

                    if (LM_lambda == 0)
                        LM_lambda = 0.2;
                    else
                        LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
                }
            }
        }

        lastResidual = lastErr;

        trackingWasGood =
            !diverged &&
            lastGoodCount / (frame->width(QUICK_KF_CHECK_LVL) * frame->height(QUICK_KF_CHECK_LVL)) > MIN_GOODPERALL_PIXEL &&
            lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

        return toSophus(referenceToFrame);
    }

    /*
        ! 跟踪新的一帧： 计算当前帧相对于参考帧的相对位姿
        主体是一个for循环，从图像金字塔的高层level-4开始遍历直到底层level-1。每一层都进行LM优化迭代，则是另外一个for循环。
        三个参数：1.参考帧 2.当前帧 3.当前帧相对于参考帧的初始位姿
    */
    // tracks a frame.
    // first_frame has depth, second_frame DOES NOT have depth.
    SE3 SE3Tracker::trackFrame(TrackingReference *reference, Frame *frame, const SE3 &frameToReference_initialEstimate)
    {
        // 
        boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
        diverged = false;
        trackingWasGood = true;
        affineEstimation_a = 1;
        affineEstimation_b = 0;

        // setting.cpp中saveAllTrackingStages默认为false
        if (saveAllTrackingStages)
        {
            saveAllTrackingStages = false;
            saveAllTrackingStagesInternal = true;
        }

        // setting.cpp中plotTrackingIterationInfo默认为false
        if (plotTrackingIterationInfo)
        {
            const float *frameImage = frame->image();
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    // getGrayCvPixel()读取该点的灰度val，返回(val,val,val)的RGB颜色值
                    // setPixelInCvMat()将该点的颜色设置为RGB：(val,val,val)
                    setPixelInCvMat(&debugImageSecondFrame, getGrayCvPixel(frameImage[col + row * width]), col, row, 1);
        }

        // !! 为跟踪当前帧设置一些变量
        // 将sophus::SE3d的初始位姿转为sophus::SE3f，也就是float类型
        Sophus::SE3f referenceToFrame = frameToReference_initialEstimate.inverse().cast<float>();
        // 最小二乘法类
        NormalEquationsLeastSquares ls;

        // setting.h中PYRAMID_LEVELS被设置为5层
        int numCalcResidualCalls[PYRAMID_LEVELS];       // 每一层计算残差(光度误差)的次数
        int numCalcWarpUpdateCalls[PYRAMID_LEVELS];     // 每一层计算最小二乘更新SE3位姿的次数

        float last_residual = 0;

        // !! 为了尺度不变性，逐层跟踪当前帧
        // setting.h中：SE3TRACKING_MAX_LEVEL=5; SE3TRACKING_MIN_LEVEL=1
        for (int lvl = SE3TRACKING_MAX_LEVEL - 1; lvl >= SE3TRACKING_MIN_LEVEL; lvl--)
        {
            numCalcResidualCalls[lvl] = 0;
            numCalcWarpUpdateCalls[lvl] = 0;

            // ***************** step1:对参考帧某一层(level)构建点云，计算了每个像素的3D空间坐标，像素梯度，颜色和方差 *****************
            reference->makePointCloud(lvl);

            // ***************** step2:计算参考点在当前帧下投影点的残差(光度误差)和梯度，并记录参考点在参考帧的逆深度和方差，论文公式13 *****************
            // 这是一个宏：#define callOptimized(function, arguments) function arguments
            // 调用宏时会执行function: 这里的calcResidualAndBuffers
            // 如果有多个参数用(): (arg1,arg2,arg3)
            callOptimized(calcResidualAndBuffers,
                          (reference->posData[lvl], reference->colorAndVarData[lvl],
                           SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame,
                           referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
            
            // 如果保存位姿的内存尺寸不够大，就返回一个零矩阵SE3
            // setting.h中MIN_GOODPERALL_PIXEL_ABSMIN=0.01
            // >> 右移位运算符： 比如width=8; lvl=2; width>>lvl = 8/(2^2)=2
            if (buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width >> lvl) * (height >> lvl))
            {
                diverged = true;
                trackingWasGood = false;
                return SE3();
            }

            // setting.h中useAffineLightningEstimation为true
            if (useAffineLightningEstimation)
            {
                affineEstimation_a = affineEstimation_a_lastIt;
                affineEstimation_b = affineEstimation_b_lastIt;
            }

            // ***************** step3:计算归一化方差的光度误差系数(论文公式14) 和 Huber-weight(论文公式15) *****************
            float lastErr = callOptimized(calcWeightsAndResidual, (referenceToFrame));
            numCalcResidualCalls[lvl]++;

            // L-M算法的lambda: 用于调整步长
            float LM_lambda = settings.lambdaInitial[lvl];
            for (int iteration = 0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
            {
                // ***************** step4/5: 计算公式12的雅可比以及最小二乘法，最后更新得到新的位姿变换SE3*****************
                callOptimized(calculateWarpUpdate, (ls));
                numCalcWarpUpdateCalls[lvl]++;

                iterationNumber = iteration;

                // ***************** step6: 不断重复step2-5，直到收敛或者到达最大迭代数 *****************
                int incTry = 0;
                while (true)
                {
                    // solve LS system with current lambda
                    Vector6 b = -ls.b;
                    Matrix6x6 A = ls.A;
                    for (int i = 0; i < 6; i++)
                        A(i, i) *= 1 + LM_lambda;
                    Vector6 inc = A.ldlt().solve(b);
                    incTry++;

                    // apply increment. pretty sure this way round is correct, but hard to test.
                    Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;
                    // Sophus::SE3f new_referenceToFrame = referenceToFrame * Sophus::SE3f::exp((inc));

                    // re-evaluate residual
                    callOptimized(calcResidualAndBuffers,
                                  (reference->posData[lvl], reference->colorAndVarData[lvl],
                                   SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl],
                                   frame, new_referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
                    if (buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width >> lvl) * (height >> lvl))
                    {
                        diverged = true;
                        trackingWasGood = false;
                        return SE3();
                    }

                    float error = callOptimized(calcWeightsAndResidual, (new_referenceToFrame));
                    numCalcResidualCalls[lvl]++;

                    // ***************** step6.1: 收敛，退出迭代计算位姿 *****************
                    if (error < lastErr)
                    {
                        // accept inc
                        referenceToFrame = new_referenceToFrame;
                        if (useAffineLightningEstimation)
                        {
                            affineEstimation_a = affineEstimation_a_lastIt;
                            affineEstimation_b = affineEstimation_b_lastIt;
                        }

                        if (enablePrintDebugInfo && printTrackingIterationInfo)
                        {
                            // debug output
                            printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n", lvl, iteration,
                                   sqrt(inc.dot(inc)), LM_lambda, lastErr, error);

                            printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f\n", referenceToFrame.log()[0], referenceToFrame.log()[1],
                                   referenceToFrame.log()[2], referenceToFrame.log()[3], referenceToFrame.log()[4],
                                   referenceToFrame.log()[5]);
                        }

                        // converged?
                        if (error / lastErr > settings.convergenceEps[lvl])
                        {
                            if (enablePrintDebugInfo && printTrackingIterationInfo)
                            {
                                printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n", lvl, iteration);
                            }
                            iteration = settings.maxItsPerLvl[lvl];
                        }

                        last_residual = lastErr = error;

                        if (LM_lambda <= 0.2)
                            LM_lambda = 0;
                        else
                            LM_lambda *= settings.lambdaSuccessFac;

                        break;
                    }
                    else
                    {
                        if (enablePrintDebugInfo && printTrackingIterationInfo)
                        {
                            printf("(%d-%d): REJECTED increment of %f with lambda %.1f, (residual: %f -> %f)\n", lvl, iteration,
                                   sqrt(inc.dot(inc)), LM_lambda, lastErr, error);
                        }

                        // ***************** step6.2: 到达最大收敛层数，退出迭代计算位姿 *****************
                        if (!(inc.dot(inc) > settings.stepSizeMin[lvl]))
                        {
                            if (enablePrintDebugInfo && printTrackingIterationInfo)
                            {
                                printf("(%d-%d): FINISHED pyramid level (stepsize too small).\n", lvl, iteration);
                            }
                            iteration = settings.maxItsPerLvl[lvl];
                            break;
                        }

                        if (LM_lambda == 0)
                            LM_lambda = 0.2;
                        else
                            LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
                    }
                }
            }
        }

        // ***************** 可视化 and 调试 *****************
        if (plotTracking)
            Util::displayImage("TrackingResidual", debugImageResiduals, false);
        if (enablePrintDebugInfo && printTrackingIterationInfo)
        {
            printf("Tracking: ");
            for (int lvl = PYRAMID_LEVELS - 1; lvl >= 0; lvl--)
            {
                printf("lvl %d: %d (%d); ", lvl, numCalcResidualCalls[lvl], numCalcWarpUpdateCalls[lvl]);
            }

            printf("\n");
        }

        saveAllTrackingStagesInternal = false;

        lastResidual = last_residual;

        // ***************** 判断当前帧跟踪的效果好不好 *****************
        trackingWasGood = !diverged &&
                          lastGoodCount / (frame->width(SE3TRACKING_MIN_LEVEL) * frame->height(SE3TRACKING_MIN_LEVEL)) >
                              MIN_GOODPERALL_PIXEL &&
                          lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;
        // 如果跟踪的好，根据当前参考帧估计位姿的帧数目+1
        if (trackingWasGood)
            reference->keyframe->numFramesTrackedOnThis++;
        
        // ***************** 保存结果到FramePoseStruct中定义的类中去 *****************
        // 平均残差(光度误差)
        frame->initialTrackedResidual = lastResidual / pointUsage;                              
        // toSophus是一个宏：#define toSophus(x) ((x).cast<double>()) 将变量转为double形式，因为前面将SE3d转成了SE3f,所以这里转回来
        // sim3FromSE3() 将三维刚体变换（SE3）转换为仿射变换（Sim3），同时设置了仿射变换的尺度scale=1。
        frame->pose->thisToParent_raw = sim3FromSE3(toSophus(referenceToFrame.inverse()), 1);
        // 当前帧跟踪的参考帧的位姿
        frame->pose->trackingParent = reference->keyframe->pose;
        // 返回的还是SE3
        return toSophus(referenceToFrame.inverse());
    }

#if defined(ENABLE_SSE)
    float SE3Tracker::calcWeightsAndResidualSSE(const Sophus::SE3f &referenceToFrame)
    {
        const __m128 txs = _mm_set1_ps((float)(referenceToFrame.translation()[0]));
        const __m128 tys = _mm_set1_ps((float)(referenceToFrame.translation()[1]));
        const __m128 tzs = _mm_set1_ps((float)(referenceToFrame.translation()[2]));

        const __m128 zeros = _mm_set1_ps(0.0f);
        const __m128 ones = _mm_set1_ps(1.0f);

        const __m128 depthVarFacs =
            _mm_set1_ps((float)settings.var_weight); // float depthVarFac = var_weight;	// the depth var is over-confident.
                                                     // this is a constant multiplier to remedy that.... HACK
        const __m128 sigma_i2s = _mm_set1_ps((float)cameraPixelNoise2);

        const __m128 huber_res_ponlys = _mm_set1_ps((float)(settings.huber_d / 2));

        __m128 sumResP = zeros;

        float sumRes = 0;

        for (int i = 0; i < buf_warped_size - 3; i += 4)
        {
            //		float px = *(buf_warped_x+i);	// x'
            //		float py = *(buf_warped_y+i);	// y'
            //		float pz = *(buf_warped_z+i);	// z'
            //		float d = *(buf_d+i);	// d
            //		float rp = *(buf_warped_residual+i); // r_p
            //		float gx = *(buf_warped_dx+i);	// \delta_x I
            //		float gy = *(buf_warped_dy+i);  // \delta_y I
            //		float s = depthVarFac * *(buf_idepthVar+i);	// \sigma_d^2

            // calc dw/dd (first 2 components):
            __m128 pzs = _mm_load_ps(buf_warped_z + i);                                          // z'
            __m128 pz2ds = _mm_rcp_ps(_mm_mul_ps(_mm_mul_ps(pzs, pzs), _mm_load_ps(buf_d + i))); // 1 / (z' * z' * d)
            __m128 g0s = _mm_sub_ps(_mm_mul_ps(pzs, txs), _mm_mul_ps(_mm_load_ps(buf_warped_x + i), tzs));
            g0s = _mm_mul_ps(g0s, pz2ds); // float g0 = (tx * pz - tz * px) / (pz*pz*d);

            // float g1 = (ty * pz - tz * py) / (pz*pz*d);
            __m128 g1s = _mm_sub_ps(_mm_mul_ps(pzs, tys), _mm_mul_ps(_mm_load_ps(buf_warped_y + i), tzs));
            g1s = _mm_mul_ps(g1s, pz2ds);

            // float drpdd = gx * g0 + gy * g1;	// ommitting the minus
            __m128 drpdds =
                _mm_add_ps(_mm_mul_ps(g0s, _mm_load_ps(buf_warped_dx + i)), _mm_mul_ps(g1s, _mm_load_ps(buf_warped_dy + i)));

            // float w_p = 1.0f / (sigma_i2 + s * drpdd * drpdd);
            __m128 w_ps = _mm_rcp_ps(_mm_add_ps(
                sigma_i2s, _mm_mul_ps(drpdds, _mm_mul_ps(drpdds, _mm_mul_ps(depthVarFacs, _mm_load_ps(buf_idepthVar + i))))));

            // float weighted_rp = fabs(rp*sqrtf(w_p));
            __m128 weighted_rps = _mm_mul_ps(_mm_load_ps(buf_warped_residual + i), _mm_sqrt_ps(w_ps));
            weighted_rps = _mm_max_ps(weighted_rps, _mm_sub_ps(zeros, weighted_rps));

            // float wh = fabs(weighted_rp < huber_res_ponly ? 1 : huber_res_ponly / weighted_rp);
            __m128 whs = _mm_cmplt_ps(weighted_rps, huber_res_ponlys); // bitmask 0xFFFFFFFF for 1, 0x000000 for
                                                                       // huber_res_ponly / weighted_rp
            whs = _mm_or_ps(_mm_and_ps(whs, ones), _mm_andnot_ps(whs, _mm_mul_ps(huber_res_ponlys, _mm_rcp_ps(weighted_rps))));

            // sumRes.sumResP += wh * w_p * rp*rp;
            if (i + 3 < buf_warped_size)
                sumResP = _mm_add_ps(sumResP, _mm_mul_ps(whs, _mm_mul_ps(weighted_rps, weighted_rps)));

            // *(buf_weight_p+i) = wh * w_p;
            _mm_store_ps(buf_weight_p + i, _mm_mul_ps(whs, w_ps));
        }
        sumRes = SSEE(sumResP, 0) + SSEE(sumResP, 1) + SSEE(sumResP, 2) + SSEE(sumResP, 3);

        return sumRes / ((buf_warped_size >> 2) << 2);
    }
#endif

#if defined(ENABLE_NEON)
    float SE3Tracker::calcWeightsAndResidualNEON(const Sophus::SE3f &referenceToFrame)
    {
        float tx = referenceToFrame.translation()[0];
        float ty = referenceToFrame.translation()[1];
        float tz = referenceToFrame.translation()[2];

        float constants[] = {
            tx, ty, tz, settings.var_weight, cameraPixelNoise2, settings.huber_d / 2,
            -1, -1 // last values are currently unused
        };
        // This could also become a constant if one register could be made free for it somehow
        float cutoff_res_ponly4[4] = {10000, 10000, 10000, 10000}; // removed
        float *cur_buf_warped_z = buf_warped_z;
        float *cur_buf_warped_x = buf_warped_x;
        float *cur_buf_warped_y = buf_warped_y;
        float *cur_buf_warped_dx = buf_warped_dx;
        float *cur_buf_warped_dy = buf_warped_dy;
        float *cur_buf_warped_residual = buf_warped_residual;
        float *cur_buf_d = buf_d;
        float *cur_buf_idepthVar = buf_idepthVar;
        float *cur_buf_weight_p = buf_weight_p;
        int loop_count = buf_warped_size / 4;
        int remaining = buf_warped_size - 4 * loop_count;
        float sum_vector[] = {0, 0, 0, 0};

        float sumRes = 0;

#ifdef DEBUG
        loop_count = 0;
        remaining = buf_warped_size;
#else
        if (loop_count > 0)
        {
            __asm__ __volatile__(
                // Extract constants
                "vldmia   %[constants], {q8-q9}              \n\t" // constants(q8-q9)
                "vdup.32  q13, d18[0]                        \n\t" // extract sigma_i2 x 4 to q13
                "vdup.32  q14, d18[1]                        \n\t" // extract huber_res_ponly x 4 to q14
                //"vdup.32  ???, d19[0]                        \n\t" // extract cutoff_res_ponly x 4 to ???
                "vdup.32  q9, d16[0]                         \n\t" // extract tx x 4 to q9, overwrite!
                "vdup.32  q10, d16[1]                        \n\t" // extract ty x 4 to q10
                "vdup.32  q11, d17[0]                        \n\t" // extract tz x 4 to q11
                "vdup.32  q8, d17[1]                         \n\t" // extract depthVarFac x 4 to q8, overwrite!

                "veor     q15, q15, q15                      \n\t" // set sumRes.sumResP(q15) to zero (by xor with itself)
                ".loopcalcWeightsAndResidualNEON:            \n\t"

                "vldmia   %[buf_idepthVar]!, {q7}           \n\t" // s(q7)
                "vldmia   %[buf_warped_z]!, {q2}            \n\t" // pz(q2)
                "vldmia   %[buf_d]!, {q3}                   \n\t" // d(q3)
                "vldmia   %[buf_warped_x]!, {q0}            \n\t" // px(q0)
                "vldmia   %[buf_warped_y]!, {q1}            \n\t" // py(q1)
                "vldmia   %[buf_warped_residual]!, {q4}     \n\t" // rp(q4)
                "vldmia   %[buf_warped_dx]!, {q5}           \n\t" // gx(q5)
                "vldmia   %[buf_warped_dy]!, {q6}           \n\t" // gy(q6)

                "vmul.f32 q7, q7, q8                        \n\t" // s *= depthVarFac
                "vmul.f32 q12, q2, q2                       \n\t" // pz*pz (q12, temp)
                "vmul.f32 q3, q12, q3                       \n\t" // pz*pz*d (q3)

                "vrecpe.f32 q3, q3                          \n\t" // 1/(pz*pz*d) (q3)
                "vmul.f32 q12, q9, q2                       \n\t" // tx*pz (q12)
                "vmls.f32 q12, q11, q0                      \n\t" // tx*pz - tz*px (q12) [multiply and subtract] {free: q0}
                "vmul.f32 q0, q10, q2                       \n\t" // ty*pz (q0) {free: q2}
                "vmls.f32 q0, q11, q1                       \n\t" // ty*pz - tz*py (q0) {free: q1}
                "vmul.f32 q12, q12, q3                      \n\t" // g0 (q12)
                "vmul.f32 q0, q0, q3                        \n\t" // g1 (q0)

                "vmul.f32 q12, q12, q5                      \n\t" // gx * g0 (q12) {free: q5}
                "vldmia %[cutoff_res_ponly4], {q5}          \n\t" // cutoff_res_ponly (q5), load for later
                "vmla.f32 q12, q6, q0                       \n\t" // drpdd = gx * g0 + gy * g1 (q12) {free: q6, q0}

                "vmov.f32 q1, #1.0                          \n\t" // 1.0 (q1), will be used later

                "vmul.f32 q12, q12, q12                     \n\t" // drpdd*drpdd (q12)
                "vmul.f32 q12, q12, q7                      \n\t" // s*drpdd*drpdd (q12)
                "vadd.f32 q12, q12, q13                     \n\t" // sigma_i2 + s*drpdd*drpdd (q12)
                "vrecpe.f32 q12, q12                        \n\t" // w_p = 1/(sigma_i2 + s*drpdd*drpdd) (q12) {free: q7}

                // float weighted_rp = fabs(rp*sqrtf(w_p));
                "vrsqrte.f32 q7, q12                        \n\t" // 1 / sqrtf(w_p) (q7)
                "vrecpe.f32 q7, q7                          \n\t" // sqrtf(w_p) (q7)
                "vmul.f32 q7, q7, q4                        \n\t" // rp*sqrtf(w_p) (q7)
                "vabs.f32 q7, q7                            \n\t" // weighted_rp (q7)

                // float wh = fabs(weighted_rp < huber_res_ponly ? 1 : huber_res_ponly / weighted_rp);
                "vrecpe.f32 q6, q7                          \n\t" // 1 / weighted_rp (q6)
                "vmul.f32 q6, q6, q14                       \n\t" // huber_res_ponly / weighted_rp (q6)
                "vclt.f32 q0, q7, q14                       \n\t" // weighted_rp < huber_res_ponly ? all bits 1 : all bits 0
                                                                  // (q0)
                "vbsl     q0, q1, q6                        \n\t" // sets elements in q0 to 1(q1) where above condition is
                                                                  // true, and to q6 where it is false {free: q6}
                "vabs.f32 q0, q0                            \n\t" // wh (q0)

                // sumRes.sumResP += wh * w_p * rp*rp
                "vmul.f32 q4, q4, q4                        \n\t" // rp*rp (q4)
                "vmul.f32 q4, q4, q12                       \n\t" // w_p*rp*rp (q4)
                "vmla.f32 q15, q4, q0                       \n\t" // sumRes.sumResP += wh*w_p*rp*rp (q15) {free: q4}

                // if(weighted_rp > cutoff_res_ponly)
                //     wh = 0;
                // *(buf_weight_p+i) = wh * w_p;
                "vcle.f32 q4, q7, q5                        \n\t" // mask in q4: ! (weighted_rp > cutoff_res_ponly)
                "vmul.f32 q0, q0, q12                       \n\t" // wh * w_p (q0)
                "vand     q0, q0, q4                        \n\t" // set q0 to 0 where condition for q4 failed (i.e.
                                                                  // weighted_rp > cutoff_res_ponly)
                "vstmia   %[buf_weight_p]!, {q0}            \n\t"

                "subs     %[loop_count], %[loop_count], #1    \n\t"
                "bne      .loopcalcWeightsAndResidualNEON     \n\t"

                "vstmia   %[sum_vector], {q15}                \n\t"

                : /* outputs */[buf_warped_z] "+&r"(cur_buf_warped_z), [buf_warped_x] "+&r"(cur_buf_warped_x),
                  [buf_warped_y] "+&r"(cur_buf_warped_y), [buf_warped_dx] "+&r"(cur_buf_warped_dx),
                  [buf_warped_dy] "+&r"(cur_buf_warped_dy), [buf_d] "+&r"(cur_buf_d),
                  [buf_warped_residual] "+&r"(cur_buf_warped_residual), [buf_idepthVar] "+&r"(cur_buf_idepthVar),
                  [buf_weight_p] "+&r"(cur_buf_weight_p), [loop_count] "+&r"(loop_count)
                : /* inputs  */[constants] "r"(constants), [cutoff_res_ponly4] "r"(cutoff_res_ponly4),
                  [sum_vector] "r"(sum_vector)
                : /* clobber */ "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
                  "q13", "q14", "q15");

            sumRes += sum_vector[0] + sum_vector[1] + sum_vector[2] + sum_vector[3];
        }
#endif

        for (int i = buf_warped_size - remaining; i < buf_warped_size; i++)
        {
            float px = *(buf_warped_x + i);                       // x'
            float py = *(buf_warped_y + i);                       // y'
            float pz = *(buf_warped_z + i);                       // z'
            float d = *(buf_d + i);                               // d
            float rp = *(buf_warped_residual + i);                // r_p
            float gx = *(buf_warped_dx + i);                      // \delta_x I
            float gy = *(buf_warped_dy + i);                      // \delta_y I
            float s = settings.var_weight * *(buf_idepthVar + i); // \sigma_d^2

            // calc dw/dd (first 2 components):
            float g0 = (tx * pz - tz * px) / (pz * pz * d);
            float g1 = (ty * pz - tz * py) / (pz * pz * d);

            // calc w_p
            float drpdd = gx * g0 + gy * g1; // ommitting the minus
            float w_p = 1.0f / (cameraPixelNoise2 + s * drpdd * drpdd);
            float weighted_rp = fabs(rp * sqrtf(w_p));

            float wh = fabs(weighted_rp < (settings.huber_d / 2) ? 1 : (settings.huber_d / 2) / weighted_rp);

            sumRes += wh * w_p * rp * rp;

            *(buf_weight_p + i) = wh * w_p;
        }

        return sumRes / buf_warped_size;
    }
#endif

    // ! 计算归一化方差的光度误差系数(论文公式14) 和 Huber-weight(论文公式15)
    float SE3Tracker::calcWeightsAndResidual(const Sophus::SE3f &referenceToFrame)
    {
        // 因为参考帧到当前帧的位姿变换比较小，所以只考虑位移t而忽略旋转R
        float tx = referenceToFrame.translation()[0];
        float ty = referenceToFrame.translation()[1];
        float tz = referenceToFrame.translation()[2];

        float sumRes = 0;

        // buf_warped_size：计算参考帧到某一帧光度误差时，用到的参考点个数
        // 计算参考帧上用到的所有参考点
        for (int i = 0; i < buf_warped_size; i++)
        {
            float px = *(buf_warped_x + i);                       // x'
            float py = *(buf_warped_y + i);                       // y'
            float pz = *(buf_warped_z + i);                       // z'
            float d = *(buf_d + i);                               // d
            float rp = *(buf_warped_residual + i);                // r_p 光度误差(残差)
            float gx = *(buf_warped_dx + i);                      // \delta_x I
            float gy = *(buf_warped_dy + i);                      // \delta_y I
            float s = settings.var_weight * *(buf_idepthVar + i); // \sigma_d^2

            // ********** 计算论文公式14的偏导数 **********
            // 公式推导：见lsd-slam笔记->代码->LSD-SLAM的跟踪->calcWeightsAndResidual()
            /*
                公式推导中 <->   代码
                dxfx     <->   gx
                dxfx后分数<->   g0
                dyfy     <->   gy
                dyfy后分数<->   g1

                gx,gy在SE3Tracker::calcResidualAndBuffers()中已经乘过焦距fx,fy了
            */ 
            // 参考点变换到当前帧后对深度的导数：
            float g0 = (tx * pz - tz * px) / (pz * pz * d);
            float g1 = (ty * pz - tz * py) / (pz * pz * d);
            // 公式14偏导数的整体：
            float drpdd = gx * g0 + gy * g1; // ommitting the minus

            // ********** 计算论文公式14的倒数**********
            float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);

            // ********** 计算论文公式15：Huber norm **********
            // 公式12：绝对值里的那个量
            float weighted_rp = fabs(rp * sqrtf(w_p));
            // 公式15: huber norm
            float wh = fabs(weighted_rp < (settings.huber_d / 2) ? 1 : (settings.huber_d / 2) / weighted_rp);

            // ********** 计算论文公式12：所有损失和 **********
            sumRes += wh * w_p * rp * rp;

            // 加了huber weight(norm)的14式
            *(buf_weight_p + i) = wh * w_p;
        }

        // 返回平均光度误差损失
        return sumRes / buf_warped_size;
    }

    // ! 开始计算变换得到的当前帧的残差和梯度：debugStart
    void SE3Tracker::calcResidualAndBuffers_debugStart()
    {
        if (plotTrackingIterationInfo || saveAllTrackingStagesInternal)
        {
            // 如果saveAllTrackingStagesInternal为真，other=255
            int other = saveAllTrackingStagesInternal ? 255 : 0;
            // 如果other为255，就是把下面这4个图像的每个像素都设置为白色
            fillCvMat(&debugImageResiduals, cv::Vec3b(other, other, 255));
            fillCvMat(&debugImageWeights, cv::Vec3b(other, other, 255));
            fillCvMat(&debugImageOldImageSource, cv::Vec3b(other, other, 255));
            fillCvMat(&debugImageOldImageWarped, cv::Vec3b(other, other, 255));
        }
    }
    
    // ! 结束计算变换得到的当前帧的残差和梯度：debugFinish
    void SE3Tracker::calcResidualAndBuffers_debugFinish(int w)
    {
        if (plotTrackingIterationInfo)
        {
            Util::displayImage("Weights", debugImageWeights);
            Util::displayImage("second_frame", debugImageSecondFrame);
            Util::displayImage("Intensities of second_frame at transformed positions", debugImageOldImageSource);
            Util::displayImage("Intensities of second_frame at pointcloud in first_frame", debugImageOldImageWarped);
            Util::displayImage("Residuals", debugImageResiduals);

            // wait for key and handle it
            bool looping = true;
            while (looping)
            {
                int k = Util::waitKey(1);
                if (k == -1)
                {
                    if (autoRunWithinFrame)
                        break;
                    else
                        continue;
                }

                char key = k;
                if (key == ' ')
                    looping = false;
                else
                    handleKey(k);
            }
        }

        if (saveAllTrackingStagesInternal)
        {
            char charbuf[500];

            snprintf(charbuf, 500, "save/%sresidual-%d-%d.png", packagePath.c_str(), w, iterationNumber);
            cv::imwrite(charbuf, debugImageResiduals);

            snprintf(charbuf, 500, "save/%swarped-%d-%d.png", packagePath.c_str(), w, iterationNumber);
            cv::imwrite(charbuf, debugImageOldImageWarped);

            snprintf(charbuf, 500, "save/%sweights-%d-%d.png", packagePath.c_str(), w, iterationNumber);
            cv::imwrite(charbuf, debugImageWeights);

            printf("saved three images for lvl %d, iteration %d\n", w, iterationNumber);
        }
    }

#if defined(ENABLE_SSE)
    float SE3Tracker::calcResidualAndBuffersSSE(const Eigen::Vector3f *refPoint, const Eigen::Vector2f *refColVar,
                                                int *idxBuf, int refNum, Frame *frame, const Sophus::SE3f &referenceToFrame,
                                                int level, bool plotResidual)
    {
        return calcResidualAndBuffers(refPoint, refColVar, idxBuf, refNum, frame, referenceToFrame, level, plotResidual);
    }
#endif

#if defined(ENABLE_NEON)
    float SE3Tracker::calcResidualAndBuffersNEON(const Eigen::Vector3f *refPoint, const Eigen::Vector2f *refColVar,
                                                 int *idxBuf, int refNum, Frame *frame,
                                                 const Sophus::SE3f &referenceToFrame, int level, bool plotResidual)
    {
        return calcResidualAndBuffers(refPoint, refColVar, idxBuf, refNum, frame, referenceToFrame, level, plotResidual);
    }
#endif

    // ! 计算参考点在当前帧下投影点的残差(光度误差)和梯度，并记录参考点在参考帧的逆深度和方差，论文公式13
    float SE3Tracker::calcResidualAndBuffers(const Eigen::Vector3f *refPoint, const Eigen::Vector2f *refColVar, int *idxBuf,
                                             int refNum, Frame *frame, const Sophus::SE3f &referenceToFrame, int level,
                                             bool plotResidual)
    {
        // ********** 开始残差梯度计算，将debug的4个图像()的每个像素都设为白色。**********
        calcResidualAndBuffers_debugStart();

        // 是否可视化残差
        if (plotResidual)
            debugImageResiduals.setTo(0);

        // 读取相机内参
        int w = frame->width(level);
        int h = frame->height(level);
        Eigen::Matrix3f KLvl = frame->K(level);
        float fx_l = KLvl(0, 0);
        float fy_l = KLvl(1, 1);
        float cx_l = KLvl(0, 2);
        float cy_l = KLvl(1, 2);

        // 读取参考帧到当前帧的位姿变换
        Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
        Eigen::Vector3f transVec = referenceToFrame.translation();

        // 最后一个参考点的地址
        const Eigen::Vector3f *refPoint_max = refPoint + refNum;
        // 当前帧的某一level的梯度
        const Eigen::Vector4f *frame_gradients = frame->gradients(level);

        // 后面对参考点操作用到的一些变量：
        int idx = 0;
        float sumResUnweighted = 0;
        bool *isGoodOutBuffer = idxBuf != 0 ? frame->refPixelWasGood() : 0;
        int goodCount = 0;
        int badCount = 0;
        float sumSignedRes = 0;
        float sxx = 0, syy = 0, sx = 0, sy = 0, sw = 0;
        float usageCount = 0;

        // 对所有参考点进行操作
        for (; refPoint < refPoint_max; refPoint++, refColVar++, idxBuf++)
        {   
            // ********** 计算参考点在当前帧坐标系下的2D坐标 **********
            // 3D空间坐标
            Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
            // 2D像素坐标
            float u_new = (Wxp[0] / Wxp[2]) * fx_l + cx_l;
            float v_new = (Wxp[1] / Wxp[2]) * fy_l + cy_l;

            // step 1a: coordinates have to be in image:
            // (inverse test to exclude NANs)
            // 判断当前参考点是否投影在图像中，如果不在就将isGoodOutBuffer设为false
            if (!(u_new > 1 && v_new > 1 && u_new < w - 2 && v_new < h - 2))
            {
                if (isGoodOutBuffer != 0)
                    isGoodOutBuffer[*idxBuf] = false;
                continue;
            }
            // ********** 计算参考点的光度误差 **********
            // 通过双线性插值得到：参考点在当前帧的灰度
            Eigen::Vector3f resInterp = getInterpolatedElement43(frame_gradients, u_new, v_new, w);
            // 得到参考点在参考帧中的灰度
            float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
            // 上面计算的在当前帧的灰度
            float c2 = resInterp[2];
            // 作差得到光度误差，也就是残差
            float residual = c1 - c2;

            // ********** 根据光度误差计算权重 **********
            // fabsf(x)返回x的绝对值
            // 光度误差<5时=1， >5时越大权重越小
            float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
            sxx += c1 * c1 * weight;
            syy += c2 * c2 * weight;
            sx += c1 * weight;
            sy += c2 * weight;
            sw += weight;

            // ********** 根据光度误差判断该参考点是不是一个好的参考点 **********
            // 如果光度误差的平方 < 下面这一串分母，就认为这个参考点是好的
            bool isGood =
                residual * residual /
                    (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT * (resInterp[0] * resInterp[0] + resInterp[1] * resInterp[1])) <
                1;
            if (isGoodOutBuffer != 0)
                isGoodOutBuffer[*idxBuf] = isGood;

            // ********** 记录参考帧上的参考点在当前帧下的一些数值 **********
            // 参考点在当前帧坐标系下的位姿T
            *(buf_warped_x + idx) = Wxp(0);
            *(buf_warped_y + idx) = Wxp(1);
            *(buf_warped_z + idx) = Wxp(2);
            // 参考点在当前帧下投影点的梯度
            *(buf_warped_dx + idx) = fx_l * resInterp[0];
            *(buf_warped_dy + idx) = fy_l * resInterp[1];
            // 参考点在当前帧下投影点的残差
            *(buf_warped_residual + idx) = residual;
            // 参考点在参考帧下的逆深度
            *(buf_d + idx) = 1.0f / (*refPoint)[2];
            // 参考点在参考帧下的方差
            *(buf_idepthVar + idx) = (*refColVar)[1];
            idx++;

            // ********** 记录光度误差的平方和 **********
            if (isGood)
            {
                sumResUnweighted += residual * residual;
                sumSignedRes += residual;
                goodCount++;
            }
            else
                badCount++;

            // ********** 记录深度改变的比例 **********
            // depthChange = 深度/Z
            // 如果深度(*refPoint)[2] < Z Wxp[2]: 那么+一个[0,1]的小数，否则加1
            float depthChange =
                (*refPoint)[2] / Wxp[2]; // if depth becomes larger: pixel becomes "smaller", hence count it less.
            usageCount += depthChange < 1 ? depthChange : 1;

            // ********** 如果设置了debug选项，就可视化原图、位姿转换后的图、光度误差图 **********
            // DEBUG STUFF
            if (plotTrackingIterationInfo || plotResidual)
            {
                // for debug plot only: find x,y again.
                // horribly inefficient, but who cares at this point...
                Eigen::Vector3f point = KLvl * (*refPoint);
                int x = point[0] / point[2] + 0.5f;
                int y = point[1] / point[2] + 0.5f;

                if (plotTrackingIterationInfo)
                {
                    setPixelInCvMat(&debugImageOldImageSource, getGrayCvPixel((float)resInterp[2]), u_new + 0.5, v_new + 0.5,
                                    (width / w));
                    setPixelInCvMat(&debugImageOldImageWarped, getGrayCvPixel((float)resInterp[2]), x, y, (width / w));
                }
                if (isGood)
                    setPixelInCvMat(&debugImageResiduals, getGrayCvPixel(residual + 128), x, y, (width / w));
                else
                    setPixelInCvMat(&debugImageResiduals, cv::Vec3b(0, 0, 255), x, y, (width / w));
            }
        }

        buf_warped_size = idx;

        pointUsage = usageCount / (float)refNum;
        lastGoodCount = goodCount;
        lastBadCount = badCount;
        lastMeanRes = sumSignedRes / goodCount;

        // ********** 计算迭代后得到的相似变换系数 **********
        affineEstimation_a_lastIt = sqrtf((syy - sy * sy / sw) / (sxx - sx * sx / sw));
        affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt * sx) / sw;

        // ********** 结束残差梯队计算 **********
        calcResidualAndBuffers_debugFinish(w);

        // 返回平均光度误差(残差)
        return sumResUnweighted / goodCount;
    }

#if defined(ENABLE_SSE)
    Vector6 SE3Tracker::calculateWarpUpdateSSE(NormalEquationsLeastSquares &ls)
    {
        ls.initialize(width * height);

        //	printf("wupd SSE\n");
        for (int i = 0; i < buf_warped_size - 3; i += 4)
        {
            Vector6 v1, v2, v3, v4;
            __m128 val1, val2, val3, val4;

            // redefine pz
            __m128 pz = _mm_load_ps(buf_warped_z + i);
            pz = _mm_rcp_ps(pz); // pz := 1/z

            __m128 gx = _mm_load_ps(buf_warped_dx + i);
            val1 = _mm_mul_ps(pz, gx); // gx / z => SET [0]
            // v[0] = z*gx;
            v1[0] = SSEE(val1, 0);
            v2[0] = SSEE(val1, 1);
            v3[0] = SSEE(val1, 2);
            v4[0] = SSEE(val1, 3);

            __m128 gy = _mm_load_ps(buf_warped_dy + i);
            val1 = _mm_mul_ps(pz, gy); // gy / z => SET [1]
            // v[1] = z*gy;
            v1[1] = SSEE(val1, 0);
            v2[1] = SSEE(val1, 1);
            v3[1] = SSEE(val1, 2);
            v4[1] = SSEE(val1, 3);

            __m128 px = _mm_load_ps(buf_warped_x + i);
            val1 = _mm_mul_ps(px, gy);
            val1 = _mm_mul_ps(val1, pz); //  px * gy * z
            __m128 py = _mm_load_ps(buf_warped_y + i);
            val2 = _mm_mul_ps(py, gx);
            val2 = _mm_mul_ps(val2, pz);   //  py * gx * z
            val1 = _mm_sub_ps(val1, val2); // px * gy * z - py * gx * z => SET [5]
            // v[5] = -py * z * gx +  px * z * gy;
            v1[5] = SSEE(val1, 0);
            v2[5] = SSEE(val1, 1);
            v3[5] = SSEE(val1, 2);
            v4[5] = SSEE(val1, 3);

            // redefine pz
            pz = _mm_mul_ps(pz, pz); // pz := 1/(z*z)

            // will use these for the following calculations a lot.
            val1 = _mm_mul_ps(px, gx);
            val1 = _mm_mul_ps(val1, pz); // px * z_sqr * gx
            val2 = _mm_mul_ps(py, gy);
            val2 = _mm_mul_ps(val2, pz); // py * z_sqr * gy

            val3 = _mm_add_ps(val1, val2);
            val3 = _mm_sub_ps(_mm_setr_ps(0, 0, 0, 0), val3); //-px * z_sqr * gx -py * z_sqr * gy
            // v[2] = -px * z_sqr * gx -py * z_sqr * gy;	=> SET [2]
            v1[2] = SSEE(val3, 0);
            v2[2] = SSEE(val3, 1);
            v3[2] = SSEE(val3, 2);
            v4[2] = SSEE(val3, 3);

            val3 = _mm_mul_ps(val1, py);                      // px * z_sqr * gx * py
            val4 = _mm_add_ps(gy, val3);                      // gy + px * z_sqr * gx * py
            val3 = _mm_mul_ps(val2, py);                      // py * py * z_sqr * gy
            val4 = _mm_add_ps(val3, val4);                    // gy + px * z_sqr * gx * py + py * py * z_sqr * gy
            val4 = _mm_sub_ps(_mm_setr_ps(0, 0, 0, 0), val4); // val4 = -val4.
            // v[3] = -px * py * z_sqr * gx +
            //       -py * py * z_sqr * gy +
            //       -gy;		=> SET [3]
            v1[3] = SSEE(val4, 0);
            v2[3] = SSEE(val4, 1);
            v3[3] = SSEE(val4, 2);
            v4[3] = SSEE(val4, 3);

            val3 = _mm_mul_ps(val1, px);   // px * px * z_sqr * gx
            val4 = _mm_add_ps(gx, val3);   // gx + px * px * z_sqr * gx
            val3 = _mm_mul_ps(val2, px);   // px * py * z_sqr * gy
            val4 = _mm_add_ps(val4, val3); // gx + px * px * z_sqr * gx + px * py * z_sqr * gy
            // v[4] = px * px * z_sqr * gx +
            //	   px * py * z_sqr * gy +
            //	   gx;				=> SET [4]
            v1[4] = SSEE(val4, 0);
            v2[4] = SSEE(val4, 1);
            v3[4] = SSEE(val4, 2);
            v4[4] = SSEE(val4, 3);

            // step 6: integrate into A and b:
            ls.update(v1, *(buf_warped_residual + i + 0), *(buf_weight_p + i + 0));

            if (i + 1 >= buf_warped_size)
                break;
            ls.update(v2, *(buf_warped_residual + i + 1), *(buf_weight_p + i + 1));

            if (i + 2 >= buf_warped_size)
                break;
            ls.update(v3, *(buf_warped_residual + i + 2), *(buf_weight_p + i + 2));

            if (i + 3 >= buf_warped_size)
                break;
            ls.update(v4, *(buf_warped_residual + i + 3), *(buf_weight_p + i + 3));
        }
        Vector6 result;

        // solve ls
        ls.finish();
        ls.solve(result);

        return result;
    }
#endif

#if defined(ENABLE_NEON)
    Vector6 SE3Tracker::calculateWarpUpdateNEON(NormalEquationsLeastSquares &ls)
    {
        //	weightEstimator.reset();
        //	weightEstimator.estimateDistributionNEON(buf_warped_residual, buf_warped_size);
        //	weightEstimator.calcWeightsNEON(buf_warped_residual, buf_warped_weights, buf_warped_size);

        ls.initialize(width * height);

        float *cur_buf_warped_z = buf_warped_z;
        float *cur_buf_warped_x = buf_warped_x;
        float *cur_buf_warped_y = buf_warped_y;
        float *cur_buf_warped_dx = buf_warped_dx;
        float *cur_buf_warped_dy = buf_warped_dy;
        Vector6 v1, v2, v3, v4;
        float *v1_ptr;
        float *v2_ptr;
        float *v3_ptr;
        float *v4_ptr;
        for (int i = 0; i < buf_warped_size; i += 4)
        {
            v1_ptr = &v1[0];
            v2_ptr = &v2[0];
            v3_ptr = &v3[0];
            v4_ptr = &v4[0];

            __asm__ __volatile__(
                "vldmia   %[buf_warped_z]!, {q10}            \n\t" // pz(q10)
                "vrecpe.f32 q10, q10                         \n\t" // z(q10)

                "vldmia   %[buf_warped_dx]!, {q11}           \n\t" // gx(q11)
                "vmul.f32 q0, q10, q11                       \n\t" // q0 = z*gx // = v[0]

                "vldmia   %[buf_warped_dy]!, {q12}           \n\t" // gy(q12)
                "vmul.f32 q1, q10, q12                       \n\t" // q1 = z*gy // = v[1]

                "vldmia   %[buf_warped_x]!, {q13}            \n\t" // px(q13)
                "vmul.f32 q5, q13, q12                       \n\t" // q5 = px * gy
                "vmul.f32 q5, q5, q10                        \n\t" // q5 = q5 * z = px * gy * z

                "vldmia   %[buf_warped_y]!, {q14}            \n\t" // py(q14)
                "vmul.f32 q3, q14, q11                       \n\t" // q3 = py * gx
                "vmls.f32 q5, q3, q10                        \n\t" // q5 = px * gy * z - py * gx * z // = v[5] (vmls: multiply
                                                                   // and subtract from result)

                "vmul.f32 q10, q10, q10                      \n\t" // q10 = 1/(pz*pz)

                "vmul.f32 q6, q13, q11                       \n\t"
                "vmul.f32 q6, q6, q10                        \n\t" // q6 = val1 in SSE version = px * z_sqr * gx

                "vmul.f32 q7, q14, q12                       \n\t"
                "vmul.f32 q7, q7, q10                        \n\t" // q7 = val2 in SSE version = py * z_sqr * gy

                "vadd.f32 q2, q6, q7                         \n\t"
                "vneg.f32 q2, q2                             \n\t" // q2 = -px * z_sqr * gx -py * z_sqr * gy // = v[2]

                "vmul.f32 q8, q6, q14                        \n\t" // val3(q8) = px * z_sqr * gx * py
                "vadd.f32 q9, q12, q8                        \n\t" // val4(q9) = gy + px * z_sqr * gx * py
                "vmul.f32 q8, q7, q14                        \n\t" // val3(q8) = py * py * z_sqr * gy
                "vadd.f32 q9, q8, q9                         \n\t" // val4(q9) = gy + px * z_sqr * gx * py + py * py * z_sqr *
                                                                   // gy
                "vneg.f32 q3, q9                             \n\t" // q3 = v[3]

                "vst4.32 {d0[0], d2[0], d4[0], d6[0]}, [%[v1]]! \n\t" // store v[0] .. v[3] for 1st value and inc pointer
                "vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [%[v2]]! \n\t" // store v[0] .. v[3] for 2nd value and inc pointer
                "vst4.32 {d1[0], d3[0], d5[0], d7[0]}, [%[v3]]! \n\t" // store v[0] .. v[3] for 3rd value and inc pointer
                "vst4.32 {d1[1], d3[1], d5[1], d7[1]}, [%[v4]]! \n\t" // store v[0] .. v[3] for 4th value and inc pointer

                "vmul.f32 q8, q6, q13                        \n\t" // val3(q8) = px * px * z_sqr * gx
                "vadd.f32 q9, q11, q8                        \n\t" // val4(q9) = gx + px * px * z_sqr * gx
                "vmul.f32 q8, q7, q13                        \n\t" // val3(q8) = px * py * z_sqr * gy
                "vadd.f32 q4, q9, q8                         \n\t" // q4 = v[4]

                "vst2.32 {d8[0], d10[0]}, [%[v1]]               \n\t" // store v[4], v[5] for 1st value
                "vst2.32 {d8[1], d10[1]}, [%[v2]]               \n\t" // store v[4], v[5] for 2nd value
                "vst2.32 {d9[0], d11[0]}, [%[v3]]               \n\t" // store v[4], v[5] for 3rd value
                "vst2.32 {d9[1], d11[1]}, [%[v4]]               \n\t" // store v[4], v[5] for 4th value

                : /* outputs */[buf_warped_z] "+r"(cur_buf_warped_z), [buf_warped_x] "+r"(cur_buf_warped_x),
                  [buf_warped_y] "+r"(cur_buf_warped_y), [buf_warped_dx] "+r"(cur_buf_warped_dx),
                  [buf_warped_dy] "+r"(cur_buf_warped_dy), [v1] "+r"(v1_ptr), [v2] "+r"(v2_ptr), [v3] "+r"(v3_ptr),
                  [v4] "+r"(v4_ptr)
                :                               /* inputs  */
                : /* clobber */ "memory", "cc", // TODO: is cc necessary?
                  "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14");

            // step 6: integrate into A and b:
            if (!(i + 3 >= buf_warped_size))
            {
                ls.update(v1, *(buf_warped_residual + i + 0), *(buf_weight_p + i + 0));
                ls.update(v2, *(buf_warped_residual + i + 1), *(buf_weight_p + i + 1));
                ls.update(v3, *(buf_warped_residual + i + 2), *(buf_weight_p + i + 2));
                ls.update(v4, *(buf_warped_residual + i + 3), *(buf_weight_p + i + 3));
            }
            else
            {
                ls.update(v1, *(buf_warped_residual + i + 0), *(buf_weight_p + i + 0));

                if (i + 1 >= buf_warped_size)
                    break;
                ls.update(v2, *(buf_warped_residual + i + 1), *(buf_weight_p + i + 1));

                if (i + 2 >= buf_warped_size)
                    break;
                ls.update(v3, *(buf_warped_residual + i + 2), *(buf_weight_p + i + 2));

                if (i + 3 >= buf_warped_size)
                    break;
                ls.update(v4, *(buf_warped_residual + i + 3), *(buf_weight_p + i + 3));
            }
        }
        Vector6 result;

        // solve ls
        ls.finish();
        ls.solve(result);

        return result;
    }
#endif

    // ! 计算公式12的雅可比以及最小二乘法，最后更新得到新的位姿变换SE3
    Vector6 SE3Tracker::calculateWarpUpdate(NormalEquationsLeastSquares &ls)
    {
        //	weightEstimator.reset();
        //	weightEstimator.estimateDistribution(buf_warped_residual, buf_warped_size);
        //	weightEstimator.calcWeights(buf_warped_residual, buf_warped_weights, buf_warped_size);
        
        // 初始化最小二乘法
        ls.initialize(width * height);
        
        // 计算参考帧上用到的所有参考点的雅可比矩阵
        for (int i = 0; i < buf_warped_size; i++)
        {
            float px = *(buf_warped_x + i);
            float py = *(buf_warped_y + i);
            float pz = *(buf_warped_z + i);
            float r  = *(buf_warped_residual + i);
            float gx = *(buf_warped_dx + i);
            float gy = *(buf_warped_dy + i);
            // step 3 + step 5 comp 6d error vector

            // ********** 计算雅可比矩阵 **********
            // 公式推导：见lsd-slam笔记->代码->LSD-SLAM的跟踪->calculateWarpUpdate()
            float z = 1.0f / pz;
            float z_sqr = 1.0f / (pz * pz);
            Vector6 v;
            v[0] = z * gx + 0;
            v[1] = 0 + z * gy;
            v[2] = (-px * z_sqr) * gx + (-py * z_sqr) * gy;
            v[3] = (-px * py * z_sqr) * gx + (-(1.0 + py * py * z_sqr)) * gy;
            v[4] = (1.0 + px * px * z_sqr) * gx + (px * py * z_sqr) * gy;
            v[5] = (-py * z) * gx + (px * z) * gy;

            // ********** 更新最小二乘问题Ax=b的A和b **********
            ls.update(v, r, *(buf_weight_p + i));
        }
        Vector6 result;

        // ********** 求解最小二乘方程 **********
        // solve ls
        ls.finish();        // 得到最后的最小二乘形式的方程
        ls.solve(result);   // 使用Eigen的ldlt求解

        return result;
    }

} // namespace lsd_slam
