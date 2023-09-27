
#======== setup.py ===========
from distutils.core import setup
from Cython.Build import cythonize

from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
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

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(Extension("render_egl",
        sources = ["renderMesh_egl.cpp", "render_egl.pyx"],
        language = "c++",
        include_dirs=incs,
        extra_link_args=libs
    )
  )
)
