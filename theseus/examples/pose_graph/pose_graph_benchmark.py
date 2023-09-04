# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pathlib

import hydra
import torch
from scipy.io import savemat

import theseus as th
import theseus.utils.examples as theg

# To run this example, you will need the cube datasets available at
# https://dl.fbaipublicfiles.com/theseus/pose_graph_data.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir datasets
#   cd datasets
#   cp your/path/pose_graph_data.tar.gz .
#   tar -xzvf pose_graph_data.tar.gz
#   cd ..
#   python examples/pose_graph_benchmark.py

# Logger
log = logging.getLogger(__name__)


# DATASET_DIR = pathlib.Path.cwd() / "datasets" / "pose_graph"
DATASET_DIR = pathlib.Path.cwd() / "datasets"

"""
Hydra 提供的装饰器,用于帮助配置管理和参数解析。需要额外的下载hydra库
hydra,main():将下面的函数,这里的main(),标记为一个可以从命令行运行的hydra应用
config_path:指定配置文件目录 config_name:制定配置文件名字
cfg是DictConfig类型
"""
@hydra.main(config_path="../configs/pose_graph", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    # file_path = f"{DATASET_DIR}/{dataset_name}_init.g2o"
    file_path = f"{DATASET_DIR}/{dataset_name}.g2o"     # f""字符串格式化。
    
    # eval()将一个字符串作为参数，然后解释并执行这个字符串表示的Python表达式或语句
    # 所以相当于执行了：dtype = torch.float64
    dtype = eval(f"torch.{cfg.dtype}")                  # cfg.dtype：float64 
    
    # 从g2o文件中读取顶点和边的信息
    # 返回值为列表类型：顶点个数，顶点，边
    _, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=torch.float64) 
    d = 3

    # th.Objective()将cost function和cost weights整合定义为一个优化问题，即构建最小二乘法问题
    # dtype=torch.float64 将所有变量的数据类型定义为torch.float64
    objective = th.Objective(torch.float64)

    # 将cost function的第一项,各位姿之间的边,放入优化目标objective
    for edge in edges:
        # th.Between() 继承了CostFunction一个类。
        cost_func = th.Between(
            verts[edge.i],      # 数据结构LieGroup类，顶点1
            verts[edge.j],      # 数据结构LieGroup类，顶点2
            edge.relative_pose, # 数据结构LieGroup类，两个顶点之间的差值，这里是相对位姿
            edge.weight,        # 数据结构CostWeight类，信息矩阵(inverse covariance协方差矩阵的逆)
        )
        objective.add(cost_func)# 将cost function即误差项加入优化目标

    # 把cost function的第二项，
    pose_prior = th.Difference(
        var=verts[0],   # 数据结构LieGroup类，位姿
        cost_weight=th.ScaleCostWeight(torch.tensor(1e-6, dtype=torch.float64)),    # # 数据结构CostWeight类，权重为1e-6
        target=verts[0].copy(new_name=verts[0].name + "PRIOR"), # 数据结构LieGroup类，目标位姿
    )
    objective.add(pose_prior)

    objective.to(dtype) # 把objective中所有的数据的结构都改成dtype: float64
    
    # 将优化器设置为L-M
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=10,
        step_size=1,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    # 这行代码是用于创建一个CUDA事件对象，通常用于测量GPU操作的时间。
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # inputs是个字典类型，存储所有的观测数据
    inputs = {var.name: var.tensor for var in verts}
    # 存入所有inputs里的变量，这里也就是顶点(观测到的位姿)
    optimizer.objective.update(inputs)

    # 将当前时间记录到 start_event 中
    start_event.record()
    # 重置 GPU 内存的峰值使用统计信息。这个函数可以在进行 GPU 内存性能分析时非常有用
    torch.cuda.reset_peak_memory_stats()
    # 使用上面定义好的L-M优化器来优化位姿图
    optimizer.optimize(verbose=True)
    # 将当前时间记录到 end_event 中
    end_event.record()

    torch.cuda.synchronize()                                    # 用于同步 CPU 和 GPU，确保前面的 GPU 操作在继续执行之前已经完成。
    forward_time = start_event.elapsed_time(end_event)          # 计算了start_event和end_event之间的时间差，获得gpu操作的执行时间
    forward_mem = torch.cuda.max_memory_allocated() / 1048576   # 计算GPU在某个操作期间分配的最大内存使用量，1MB=1048576字节Byte
    log.info(f"Forward pass took {forward_time} ms.")
    log.info(f"Forward pass used {forward_mem} MBs.")

    results = {}
    # 默认的error_metric()是评估误差项的squared norm然后除以2
    results["objective"] = objective.error_metric().detach().cpu().numpy().sum()  
    # 从所有顶点也即4x4的变换矩阵中将3x3的旋转矩阵提取出来，注意这里0,0,0那一行也会被读取出来
    # d上面定义了3，所以取0,1,2这前3列
    results["R"] = torch.cat(
        [pose.tensor[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    # 去除变换的那个部分
    results["t"] = torch.cat(
        [pose.tensor[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
