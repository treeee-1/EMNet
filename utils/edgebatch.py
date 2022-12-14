import os
import cv2 as cv

path = 'SegmentationClassout/'
# 获取rootdir目录下的文件名清单
list = os.listdir(path)

for i in range(0, len(list)):  # 遍历目录下的所有文件夹
    dir_path = os.path.join(path, list[i])  # 文件夹的路径
    dir_name = list[i]  # 获得此文件夹的名字


    def edge_demo(dir_name):
        src1 = cv.imread(path + dir_name)
        blurred = cv.GaussianBlur(src1, (3, 3), 0)
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
        ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
        edge_output = cv.Canny(xgrad, ygrad, 40, 100)
        #
        # Img_Name = "UAVidDataset/edgelabel/" + str(i+1) + ".png"
        # # if not os.path.exists(save_path):
        # #     os.makedirs(save_path)
        # cv.imwrite(Img_Name, edge_output)
# ====================================================
#     分别创建跟文件同名的文件夹，并将结果保存在同名文件夹里
#         save_path = path + dir_name + "edge{0}/".format(i+1)
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#
#         cv.imwrite(save_path + 'Canny_Edge.png', edge_output)
# ====================================================
#     将所有结果保存到一个指定文件夹里
        dir = dir_name.split('.')[0]
        save_path = 'SegmentationClassout1/'
        cv.imwrite(save_path + f'Canny_Edge{dir}.png', edge_output)

    edge_demo(dir_name)
