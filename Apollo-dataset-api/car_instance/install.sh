#!/bin/bash
# make sure pip and setuptools are available on your system

cd renderer/
# build_ext用于构建Python扩展模块。允许你在Python中使用底层的C/C++代码
# 参数：-i 或 --inplace：将编译后的扩展模块直接放置在源文件所在目录中。
python3 setup.py build_ext --inplace
curr_dir=`pwd`
cd $curr_dir
