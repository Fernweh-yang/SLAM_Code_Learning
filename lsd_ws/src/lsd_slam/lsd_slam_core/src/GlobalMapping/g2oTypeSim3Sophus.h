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
// ! 这个文件定义了g2o图优化所需要的顶点g2o::BaseVertex<7, Sophus::Sim3d> 和 边g2o::BaseBinaryEdge<7, Sophus::Sim3d, VertexSim3, VertexSim3>
#pragma once
#include "util/SophusUtil.h"

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

namespace lsd_slam
{
    // 重写顶点：顶点的维度为7，数据类型是sophus中定义的sim3d
    // Sim(3)三维相似变换矩阵：因为单目相机的尺度不确定，所以
    class VertexSim3 : public g2o::BaseVertex<7, Sophus::Sim3d>
    {
    public:
        // EIGEN的宏，确保操作符分配的内存是对齐的
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        // 构造函数
        VertexSim3();
        
        // read 和 write 函数分别用于从输入流读取顶点信息和将顶点信息写入输出流。
        virtual bool read(std::istream &is);
        virtual bool write(std::ostream &os) const;
        
        // 顶点的重置函数
        virtual void setToOriginImpl()
        {
            _estimate = Sophus::Sim3d();    // 要被优化的估计值的初始值设为Sim3d()，其中每个元素为0
        }

        // 顶点的更新函数,也就是位姿的更新
        virtual void oplusImpl(const double *update_)
        {
            // const_cast用于去除update_变量的常量修饰符const
            // 从而允许将其存储到非常量变量：update
            Eigen::Map<Eigen::Matrix<double, 7, 1>> update(const_cast<double *>(update_));

            if (_fix_scale)
                update[6] = 0;  // 如果_fix_scale为真，将第7个元素也就是尺度设置为0

            // Sophus::Sim3d::exp(update) 将增量变换转换为sim3变换
            // 然后乘以当前的估计值：estimate()
            // 最后通过setEstimate()将结果设置为新的估计值 
            setEstimate(Sophus::Sim3d::exp(update) * estimate());
        }

        bool _fix_scale;
    };

    /**
     * \brief 7D edge between two Vertex7
     */
    // 重写边：主要写自己的误差模型
    class EdgeSim3 : public g2o::BaseBinaryEdge<7, Sophus::Sim3d, VertexSim3, VertexSim3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // 构造函数
        EdgeSim3();

        // read 和 write 函数分别用于从输入流读取顶点信息和将顶点信息写入输出流。
        virtual bool read(std::istream &is);
        virtual bool write(std::ostream &os) const;

        // 重写误差函数
        void computeError()
        {   
            // 第一个顶点，位姿0
            const VertexSim3 *_from = static_cast<const VertexSim3 *>(_vertices[0]);
            // 第二个顶点，位姿1
            const VertexSim3 *_to = static_cast<const VertexSim3 *>(_vertices[1]);
            // 计算误差：位姿1的逆 * 位姿2 * 边的逆
            // ? 类似于14讲271页公式10.4 
            Sophus::Sim3d error_ = _from->estimate().inverse() * _to->estimate() * _inverseMeasurement;
            _error = error_.log();
        }

        // 雅可比矩阵计算
        void linearizeOplus()
        {
            const VertexSim3 *_from = static_cast<const VertexSim3 *>(_vertices[0]);

            _jacobianOplusXj = _from->estimate().inverse().Adj();
            _jacobianOplusXi = -_jacobianOplusXj;
        }

        virtual void setMeasurement(const Sophus::Sim3d &m)
        {
            _measurement = m;
            _inverseMeasurement = m.inverse();
        }

        virtual bool setMeasurementData(const double *m)
        {
            // Eigen::Map<g2o::Vector7> test;
            Eigen::Map<const g2o::Vector7> v(m);
            setMeasurement(Sophus::Sim3d::exp(v));
            return true;
        }

        virtual bool setMeasurementFromState()
        {
            const VertexSim3 *from = static_cast<const VertexSim3 *>(_vertices[0]);
            const VertexSim3 *to = static_cast<const VertexSim3 *>(_vertices[1]);
            Sophus::Sim3d delta = from->estimate().inverse() * to->estimate();
            setMeasurement(delta);
            return true;
        }

        virtual double initialEstimatePossible(const g2o::OptimizableGraph::VertexSet &, g2o::OptimizableGraph::Vertex *)
        {
            return 1.;
        }

        virtual void initialEstimate(const g2o::OptimizableGraph::VertexSet &from, g2o::OptimizableGraph::Vertex * /*to*/)
        {
            VertexSim3 *_from = static_cast<VertexSim3 *>(_vertices[0]);
            VertexSim3 *_to = static_cast<VertexSim3 *>(_vertices[1]);

            if (from.count(_from) > 0)
                _to->setEstimate(_from->estimate() * _measurement);
            else
                _from->setEstimate(_to->estimate() * _inverseMeasurement);
        }

    protected:
        Sophus::Sim3d _inverseMeasurement;
    };

} // namespace lsd_slam
