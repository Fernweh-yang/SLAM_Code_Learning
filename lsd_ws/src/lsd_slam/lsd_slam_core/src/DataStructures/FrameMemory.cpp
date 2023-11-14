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

#include "DataStructures/FrameMemory.h"
#include "DataStructures/Frame.h"

namespace lsd_slam
{
    FrameMemory::FrameMemory()
    {
    }

    FrameMemory &FrameMemory::getInstance()
    {
        static FrameMemory theOneAndOnly;
        return theOneAndOnly;
    }

    void FrameMemory::releaseBuffes()
    {
        boost::unique_lock<boost::mutex> lock(accessMutex);
        int total = 0;

        for (auto p : availableBuffers)
        {
            if (printMemoryDebugInfo)
                printf("deleting %d buffers of size %d!\n", (int)p.second.size(), (int)p.first);

            total += p.second.size() * p.first;

            for (unsigned int i = 0; i < p.second.size(); i++)
            {
                Eigen::internal::aligned_free(p.second[i]);
                bufferSizes.erase(p.second[i]);
            }

            p.second.clear();
        }
        availableBuffers.clear();

        if (printMemoryDebugInfo)
            printf("released %.1f MB!\n", total / (1000000.0f));
    }

    void *FrameMemory::getBuffer(unsigned int sizeInByte)
    {   
        // 调用boost里面的互斥锁，把这段函数里面的这段内存锁上了
        boost::unique_lock<boost::mutex> lock(accessMutex);
        // 如果当前可用的临时存储空间有sizeInByte大小的内存
        if (availableBuffers.count(sizeInByte) > 0)
        {   
            // 获取sizeInByte所对应的value的引用，也就是需要的内存的首地址
            std::vector<void *> &availableOfSize = availableBuffers.at(sizeInByte);
            // 如果可用的临时存储空在sizeInByte处是空的，就申请一段新的内存
            if (availableOfSize.empty())
            {   
                void *buffer = allocateBuffer(sizeInByte);
                //			assert(buffer != 0);
                return buffer;
            }
            // 如果不是空的，就直接得到sizeInByte大小的内存
            else
            {   
                void *buffer = availableOfSize.back();
                availableOfSize.pop_back();
                //			assert(buffer != 0);
                return buffer;
            }
        }
        // 如果没有
        else
        {
            void *buffer = allocateBuffer(sizeInByte);
            //		assert(buffer != 0);
            return buffer;
        }
    }

    float *FrameMemory::getFloatBuffer(unsigned int size)
    {
        return (float *)getBuffer(sizeof(float) * size);
    }

    void FrameMemory::returnBuffer(void *buffer)
    {
        if (buffer == 0)
            return;

        boost::unique_lock<boost::mutex> lock(accessMutex);

        unsigned int size = bufferSizes.at(buffer);
        // printf("returnFloatBuffer(%d)\n", size);
        if (availableBuffers.count(size) > 0)
            availableBuffers.at(size).push_back(buffer);
        else
        {
            std::vector<void *> availableOfSize;
            availableOfSize.push_back(buffer);
            availableBuffers.insert(std::make_pair(size, availableOfSize));
        }
    }

    void *FrameMemory::allocateBuffer(unsigned int size)
    {
        // printf("allocateFloatBuffer(%d)\n", size);
        // 调用eigen的内存管理函数，底层实际上还是malloc，如果失败会抛出一个throw_std_bad_alloc
        void *buffer = Eigen::internal::aligned_malloc(size);
        // 把buffer的首地址和尺寸映射起来，之后返回buffer的首地址
        bufferSizes.insert(std::make_pair(buffer, size));
        return buffer;
    }

    boost::shared_lock<boost::shared_mutex> FrameMemory::activateFrame(Frame *frame)
    {
        boost::unique_lock<boost::mutex> lock(activeFramesMutex);
        if (frame->isActive)
            activeFrames.remove(frame);
        activeFrames.push_front(frame);
        frame->isActive = true;
        return boost::shared_lock<boost::shared_mutex>(frame->activeMutex);
    }
    void FrameMemory::deactivateFrame(Frame *frame)
    {
        boost::unique_lock<boost::mutex> lock(activeFramesMutex);
        if (!frame->isActive)
            return;
        activeFrames.remove(frame);

        while (!frame->minimizeInMemory())
            printf("cannot deactivateFrame frame %d, as some acvite-lock is lingering. May cause deadlock!\n",
                   frame->id()); // do it in a loop, to make shure it is really, really deactivated.

        frame->isActive = false;
    }
    void FrameMemory::pruneActiveFrames()
    {
        boost::unique_lock<boost::mutex> lock(activeFramesMutex);

        while ((int)activeFrames.size() > maxLoopClosureCandidates + 20)
        {
            if (!activeFrames.back()->minimizeInMemory())
            {
                if (!activeFrames.back()->minimizeInMemory())
                {
                    printf("failed to minimize frame %d twice. maybe some active-lock is lingering?\n", activeFrames.back()->id());
                    return; // pre-emptive return if could not deactivate.
                }
            }
            activeFrames.back()->isActive = false;
            activeFrames.pop_back();
        }
    }

} // namespace lsd_slam
