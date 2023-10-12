
#======== setup.py ===========
from distutils.core import setup          # 用于配置Python软件包的构建和安装。
from Cython.Build import cythonize        # 用于将Cython源代码编译成Cython扩展模块。

from distutils.extension import Extension # 用于定义C/C++扩展模块的构建参数。
from Cython.Distutils import build_ext    # 用于处理Cython扩展模块的构建。
import subprocess                         # 用于在脚本中执行命令行命令。
import numpy

"""
subprocess.check_output()函数可以执行一条shell命令
pkg-config: 用来返回已安装库的基本信息
--libs: output all linker flags  获取库的链接库路径
--cflags: output all pre-processor and compiler flags 获取库的头文件路径

 subprocess.check_output() 函数,默认返回的是bytes 类型的字符串，由字节组成，默认编码是ASCII
 可以通过 text=True 参数来转换成str类型的字符串, 由Unicode字符组成, 默认编码是UTF-8
 
"""
proc_libs = subprocess.check_output("pkg-config --libs eigen3 egl glew".split(),text=True)
proc_incs = subprocess.check_output("pkg-config --cflags eigen3 egl glew".split(), text=True)

print(type(proc_libs));
print(type(proc_incs));

libs = [lib for lib in proc_libs.split()]
incs= [inc for inc in proc_incs.split()]

incs_new = []
for inc in incs:
    if '-I' in inc:
        inc = inc[2:]
    incs_new.append(inc)

incs = incs_new
incs = incs + [numpy.get_include()]
libs = libs + ['-lboost_system']

# 使用setup函数配置Python软件包的构建过程。
setup(
  cmdclass = {'build_ext': build_ext},                        # 定义build_ext命令
  ext_modules = cythonize(Extension("render_egl",             # 指定了Cython扩展模块的参数， 模块名称
        sources = ["renderMesh_egl.cpp", "render_egl.pyx"],   # 模块名称，源代码文件
        language = "c++",                                     # 语言
        include_dirs=incs,                                    # 头文件路径
        extra_link_args=libs                                  # 链接标志
    )
  )
)
