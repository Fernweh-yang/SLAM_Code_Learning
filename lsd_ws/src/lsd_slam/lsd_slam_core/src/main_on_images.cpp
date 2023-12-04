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

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <boost/thread.hpp>

#include <ros/package.h>

#include "opencv2/opencv.hpp"

#include "LiveSLAMWrapper.h"
#include "SlamSystem.h"
#include "IOWrapper/ROS/ROSOutput3DWrapper.h"
#include "IOWrapper/ROS/rosReconfigure.h"
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/Undistorter.h"

std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}
std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}
std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}
int getdir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);

        if (name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);

    std::sort(files.begin(), files.end());

    if (dir.at(dir.length() - 1) != '/')
        dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++)
    {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

int getFile(std::string source, std::vector<std::string> &files)
{
    std::ifstream f(source.c_str());

    if (f.good() && f.is_open())
    {
        while (!f.eof())
        {
            std::string l;
            std::getline(f, l);

            l = trim(l);

            if (l == "" || l[0] == '#')
                continue;

            files.push_back(l);
        }

        f.close();

        size_t sp = source.find_last_of('/');
        std::string prefix;
        if (sp == std::string::npos)
            prefix = "";
        else
            prefix = source.substr(0, sp);

        for (unsigned int i = 0; i < files.size(); i++)
        {
            if (files[i].at(0) != '/')
                files[i] = prefix + "/" + files[i];
        }

        return (int)files.size();
    }
    else
    {
        f.close();
        return -1;
    }
}

using namespace lsd_slam;
int main(int argc, char **argv)
{
    ros::init(argc, argv, "LSD_SLAM");

    // ************** 设置ros服务通信 **************
    // dynamic_reconfigure::Server是ros自带的类，用于创建一个可以动态修改参数的服务器对象
    // lsd_slam_core::LSDParamsConfig是自定义的参数配置文件，件lsd_slam_core/cfg/LSDParams.cfg
    // ("~") 表示该节点句柄指向当前节点的私有命名空间
    dynamic_reconfigure::Server<lsd_slam_core::LSDParamsConfig> srv(ros::NodeHandle("~"));
    srv.setCallback(dynConfCb);
    dynamic_reconfigure::Server<lsd_slam_core::LSDDebugParamsConfig> srvDebug(ros::NodeHandle("~Debug"));
    srvDebug.setCallback(dynConfCbDebug);

    // 得到ros某个包的路径
    packagePath = ros::package::getPath("lsd_slam_core") + "/";

    // ************** 读取相机内参 **************
    // get camera calibration in form of an undistorter object.
    // if no undistortion is required, the undistorter will just pass images through.
    std::string calibFile;
    Undistorter *undistorter = 0;
    // 从命令行输入的参数_calib中读取内参文件的地址，保存在变量calibFile中
    // 这里calibFile传进来的是内参标定文件的地址，如：/home/yang/Downloads/LSD_room_images/LSD_room/cameraCalibration.cfg
    if (ros::param::get("~calib", calibFile))
    {
        // 读取内参文件，并保存在undistorter中
        undistorter = Undistorter::getUndistorterForFile(calibFile.c_str());
        ros::param::del("~calib");
    }

    if (undistorter == 0)
    {
        printf("need camera calibration file! (set using _calib:=FILE)\n");
        exit(0);
    }

    int w = undistorter->getOutputWidth();              // 输出图像宽度：640
    int h = undistorter->getOutputHeight();             // 输出图像高度：480

    int w_inp = undistorter->getInputWidth();           // 输入图像宽度：640
    int h_inp = undistorter->getInputHeight();          // 输入图像高度：480

    float fx = undistorter->getK().at<double>(0, 0);
    float fy = undistorter->getK().at<double>(1, 1);
    float cx = undistorter->getK().at<double>(2, 0);
    float cy = undistorter->getK().at<double>(2, 1);
    Sophus::Matrix3f K;                                 // 内参矩阵         
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    // ************** 设置可视化 **************
    // make output wrapper. just set to zero if no output is required.
    Output3DWrapper *outputWrapper = new ROSOutput3DWrapper(w, h);

    // ************** 创建lsd-slam系统 **************
    // make slam system
    // doslam在setting.cpp中默认为true
    SlamSystem *system = new SlamSystem(w, h, K, doSlam);
    system->setVisualization(outputWrapper);

    // ************** 读取数据集照片 **************
    // open image files: first try to open as file.
    std::string source;
    std::vector<std::string> files;
    if (!ros::param::get("~files", source))
    {
        printf("need source files! (set using _files:=FOLDER)\n");
        exit(0);
    }
    ros::param::del("~files");

    if (getdir(source, files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)files.size(), source.c_str());
    }
    else if (getFile(source, files) >= 0)
    {
        printf("found %d image files in file %s!\n", (int)files.size(), source.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    // 从命令行输入的参数_hz中读取处理图像的频率
    double hz = 0;
    if (!ros::param::get("~hz", hz))
        hz = 0;
    ros::param::del("~hz");

    cv::Mat image = cv::Mat(h, w, CV_8U);
    int runningIDX = 0;
    float fakeTimeStamp = 0;

    // ros::Rate 确保ros节点中的循环以给定的频率运行
    ros::Rate r(hz);

    for (unsigned int i = 0; i < files.size(); i++)
    {   
        // ************** 读取数据集每一帧图像，并进行内参矫正 **************
        cv::Mat imageDist = cv::imread(files[i], cv::IMREAD_GRAYSCALE);

        if (imageDist.rows != h_inp || imageDist.cols != w_inp)
        {
            if (imageDist.rows * imageDist.cols == 0)
                printf("failed to load image %s! skipping.\n", files[i].c_str());
            else
                printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n", files[i].c_str(), w, h,
                       imageDist.cols, imageDist.rows);
            continue;
        }
        assert(imageDist.type() == CV_8U);

        // ************** 对每一帧进行内参矫正，并保存在image中 **************
        undistorter->undistort(imageDist, image);
        assert(image.type() == CV_8U);

        // ************** Tracking **************
        if (runningIDX == 0)
            // 启动lsd-slam时，给第一个关键帧任意初始化一个深度地图和方差
            // opencv::Mat的data属性返回一个指向Mat中数据的指针
            system->randomInit(image.data, fakeTimeStamp, runningIDX);
        else
            // 启动后，就是追踪每一帧
            system->trackFrame(image.data, runningIDX, hz == 0, fakeTimeStamp);
        runningIDX++;
        fakeTimeStamp += 0.03;

        if (hz != 0)
            r.sleep();  // 确保ros节点中的循环以给定的频率运行

        // 重置slam系统，在setting.cpp中默认为false，手动输入R会变为true
        if (fullResetRequested)
        {
            printf("FULL RESET!\n");
            delete system;

            system = new SlamSystem(w, h, K, doSlam);
            system->setVisualization(outputWrapper);

            fullResetRequested = false;
            runningIDX = 0;
        }

        ros::spinOnce();

        if (!ros::ok())
            break;
    }

    system->finalize();

    delete system;
    delete undistorter;
    delete outputWrapper;
    return 0;
}
