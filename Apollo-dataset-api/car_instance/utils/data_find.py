"""
用于寻找car instance和其他数据集中相同的图片
"""
import os
import numpy as np

# 获取指定目录下的所有文件夹
def get_all_folders(folder_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        folders.extend(dirnames)
        return folders

if __name__ == "__main__":    
    # ****************** 获取car_instance目录下所有照片文件的名字 ******************
    # 获取当前工作目录
    current_directory = os.getcwd()
    current_directory = '/media/yang/OS/Datasets/ApolloScape/CarInstance/train/images'
    # 使用 os.listdir() 列出当前目录下的所有文件和子目录
    files_and_directories = os.listdir(current_directory)

    # 筛选出当前目录下的所有文件（去除子目录）
    # os.path.join() 函数用于将多个路径组合成一个路径：os.path.join("data", "images", "cat.jpg")=>data/images/cat.jpg
    files = [f for f in files_and_directories if os.path.isfile(os.path.join(current_directory, f))]

    # 将list转为np.array
    array_car_instance = np.array(files)
    print('171206_034625454_Camera_5.jpg' in array_car_instance)

    # ****************** 获取scene_parsing目录下所有的照片文件名字 ******************
    ROOT = '/media/yang/OS/Datasets/ApolloScape/scene_parsing/'
    dataset_folders = ['road01_ins','road02_ins','road03_ins','road04_ins']
    camera_folders = ['Camera 5', 'Camera 6']
    num_match = 0
    for item_1 in dataset_folders:
        folders_path_1 = ROOT + '%s/ColorImage/' %item_1
        folder = get_all_folders(folders_path_1)
        for item_2 in folder:
            folders_path_2 = folders_path_1 + '%s/' %item_2
            for item_3 in camera_folders:
                folders_path_3 = folders_path_2 + '%s' %item_3
                # print(folders_path_3)
                files2search = os.listdir(folders_path_3)
                files = [f for f in files2search if os.path.isfile(os.path.join(folders_path_3, f))]
                for item in files:
                    if item in array_car_instance:
                        num_match = num_match +1
                        print(item)
    print(num_match)
    

