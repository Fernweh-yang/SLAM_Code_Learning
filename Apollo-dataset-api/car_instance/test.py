import sys
sys.path.insert(0, './renderer')
sys.path.insert(0, '../')

"""
PyTorch 的 collections 库提供了一些用于管理和操作 Python 集合的工具。
namedtuple: 用于创建具有命名属性的元组。
"""
from collections import namedtuple
import render_car_instances as rci
import utils.utils as uts

Setting = namedtuple('Setting', ['image_name', 'data_dir'])
setting = Setting('180116_053947113_Camera_5', '../apolloscape/sample/')
# /home/yang/Datasets/ApolloScape/CarInstance/sample/car_models//baojun-310-2017.pkl
# setting = Setting('180116_053947113_Camera_5', '/home/yang/Datasets/ApolloScape/CarInstance/sample/')


"""
matplotlib inline是IPython的魔法函数,
可以在IPython编译器里直接使用,作用是内嵌画图,省略掉plt.show()这一步，直接显示图像。
"""
# %matplotlib inline
visualizer = rci.CarPoseVisualizer(setting)
# print(visualizer.dataset)
# visualizer.load_car_models()
# image_vis, mask, depth = visualizer.showAnn(setting.image_name)