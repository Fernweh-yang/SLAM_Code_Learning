import cv2
import os
from PIL import Image

images_path = 'C:\\Users\\51212\\Desktop\\testeverything\\nerf\\CodeComments_NeEF\\data\\nerf_llff_data\\drone\\images\\' # 原图路径
output_dir = 'C:\\Users\\51212\\Desktop\\testeverything\\nerf\\CodeComments_NeEF\\data\\nerf_llff_data\\drone\\images_8\\' # resize后路径

factor = 8 # 降采样倍数

images_list = os.listdir(images_path)
x = images_path + images_list[0]
img = Image.open(images_path + images_list[0])
(W,H) = (img.width,img.height) #[W,H]
print("image_size : ",(W ,H))

for image_name in images_list:
    img = cv2.imread(images_path+image_name)
    img_resize = cv2.resize(img, (int(W/factor), int(H/factor)))
    cv2.imwrite(output_dir + image_name, img_resize)
    print(image_name , " done")
print("all images done")