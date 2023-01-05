from keras.models import *
from keras.layers import *
import os
import numpy as np
import math
from skimage import io, transform
from pickle import dump
from skimage.color import rgb2gray
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from os import listdir
import copy
import cv2


def HilbertCoordinates(cali_read, img_read):
    """
    读取图像矩阵，根据希尔伯特曲线展开思想，取得图像展开坐标
    """
    # 二类间盘标定矩阵归一化
    cali_read[cali_read < 20] = 1

    # 去除小像素点
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(cali_read, kernel, iterations=1)
    canny_data = cv2.Canny(dilate, 1, 0)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(canny_data)
    i = 0
    for istat in stats:
        if istat[4] < 45:  # 过滤阈值
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(canny_data, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
        i = i + 1
    # max_data = np.max(cali_read)
    # print('输入点云共', max_data, '类')
    # cv2.imwrite('./tmp/tmp.png', canny_data)

    # 确定左右边界
    for i_w in range(canny_data.shape[1]):
        if sum(canny_data[:, i_w]) != 0:
            cali_iw = copy.deepcopy(i_w)
            break
    for i_e in range(canny_data.shape[1]):
        if sum(canny_data[:, (canny_data.shape[1] - i_e - 1)]) != 0:
            cali_ie = copy.deepcopy(canny_data.shape[1] - i_e - 1)
            break
    # print('左边界', cali_iw, '右边界', cali_ie)

    # 确定上下单元边界
    matrix_ns = np.zeros(10)
    # print(matrix_ns.shape[0])
    n = matrix_ns.shape[0]

    i_mc = 0
    for i_s in range(canny_data.shape[0]):
        if i_mc < n:
            if bool(sum(canny_data[(canny_data.shape[0] - i_s - 1), :])) != bool(
                    sum(canny_data[(canny_data.shape[0] - i_s - 2), :])):
                matrix_ns[i_mc] = canny_data.shape[0] - i_s - 1
                i_mc = i_mc + 1
    # print('上下边界倒序',matrix_ns)
    # matrix_ns = matrix_ns + 5
    # print(matrix_ns)

    # 依照希尔伯特曲线展开思想获取对于图像坐标
    Ra_ImgCali_ew = cali_read.shape[1] / img_read.shape[1]
    Ra_ImgCali_sn = cali_read.shape[0] / img_read.shape[0]
    # print('行比例：', Ra_ImgCali_ew)
    # print('列比例：', Ra_ImgCali_sn)

    matrix_ns = matrix_ns / Ra_ImgCali_sn
    img_iw = int(round(cali_iw / Ra_ImgCali_ew))
    img_ie = int(round(cali_ie / Ra_ImgCali_ew))
    # print('图像行坐标：', img_iw, '&', img_ie)
    # print('图像列坐标：', matrix_ns)

    return matrix_ns, img_iw, img_ie


def extract_features(directory):

    model = VGG16()  # 加载模型VGG16
    model.layers.pop()  # 去掉最后一层
    model.summary()  # 输出summarize
    features = {}  # 从图片中提取特征,先构建一个空字典

    coordinate_allcase_files = []
    patient_allcase_folders = []

    for root, dirs, files in os.walk('G:\MRI_imagecaptain\disc_coordinate_binary'):
        for name in files:
            files_root = os.path.join(root, name)
            coordinate_allcase_files.append(files_root)
    coordinate_files_len = len(coordinate_allcase_files)

    for root_d, dirs_d, files_d in os.walk(directory):
        for fdir in dirs_d:
            folder_root = os.path.join(root_d, fdir)
            # print(folder_root)
            patient_allcase_folders.append(folder_root)

    for files_num in range(0, coordinate_files_len):
        print('第', str(files_num + 1), '位患者的间盘坐标开始处理')
        coordinate_onecase_files = coordinate_allcase_files[files_num]
        #print(coordinate_onecase_files)
        #print(patient_allcase_folders[files_num])
        cali_data = np.array(cv2.imread(coordinate_onecase_files, cv2.IMREAD_GRAYSCALE))
        # print(patient_allcase_folders)
        imgs_in_patient = []
        for root, dirs, files in os.walk(patient_allcase_folders[files_num]):
            for file in files:
                img_root = os.path.join(root, file)
                #print(img_root)
                imgs_in_patient.append(img_root)
            over_num_ = True
            if len(imgs_in_patient) < 8:
                while over_num_:
                    imgs_in_patient.append(img_root)
                    if len(imgs_in_patient) == 8:
                        over_num_ = False

        imgs_in_patient_len = len(imgs_in_patient)
        fris_img_data = np.array(cv2.imread(imgs_in_patient[5], cv2.IMREAD_GRAYSCALE))
        matrix_ns, img_iw, img_ie = HilbertCoordinates(cali_data, fris_img_data)
        # new_matrix_all = np.zeros((math.ceil(matrix_ns[0] + 10)-math.ceil(matrix_ns[-1] - 10), (img_ie -img_iw + 30)))
        # print('new_matrix_all.shape')
        # print(new_matrix_all.shape)

        over_num = False
        while over_num:
            if imgs_in_patient_len == 8:
                over_num = True
            elif imgs_in_patient_len == 9:
                imgs_in_patient.pop(0)
                over_num = False
            elif imgs_in_patient_len > 9:
                imgs_in_patient.pop(0)
                imgs_in_patient.pop(imgs_in_patient_len - 2)
                over_num = False

        new_matrix_all = np.zeros((math.ceil(matrix_ns[0] + 10)-(math.ceil(matrix_ns[1] - 10)), (img_ie + 10 - img_iw + 20), 3))
        for img_num in range(3, 6):
            print('第'+str(img_num))
            patient_img = imgs_in_patient[img_num]
            fris_img_data = np.array(cv2.imread(patient_img, cv2.IMREAD_COLOR))

            new_matrix = fris_img_data[math.ceil(matrix_ns[1] - 10):math.ceil(matrix_ns[0] + 10), img_iw - 20:img_ie + 10]

            new_matrix_all = new_matrix_all + np.array(new_matrix)
        new_matrix_all = new_matrix_all / 3
        cv2.imwrite('dices_all/' + patient_allcase_folders[files_num][32:] + '_L5_S1.png', new_matrix_all)

        new_matrix_all = np.zeros((math.ceil(matrix_ns[2] + 10)-(math.ceil(matrix_ns[3] - 10)), (img_ie + 10 - img_iw + 20), 3))
        for img_num in range(3, 6):
            print('第'+str(img_num))
            patient_img = imgs_in_patient[img_num]
            fris_img_data = np.array(cv2.imread(patient_img, cv2.IMREAD_COLOR))

            new_matrix = fris_img_data[math.ceil(matrix_ns[3] - 10):math.ceil(matrix_ns[2] + 10), img_iw - 20:img_ie + 10]

            new_matrix_all = new_matrix_all + np.array(new_matrix)
        new_matrix_all = new_matrix_all / 3
        cv2.imwrite('dices_all/' + patient_allcase_folders[files_num][32:] + '_L4_L5.png', new_matrix_all)

        new_matrix_all = np.zeros((math.ceil(matrix_ns[4] + 10)-(math.ceil(matrix_ns[5] - 10)), (img_ie + 10 - img_iw + 20), 3))
        for img_num in range(2, 6):
            print('第'+str(img_num))
            patient_img = imgs_in_patient[img_num]
            fris_img_data = np.array(cv2.imread(patient_img, cv2.IMREAD_COLOR))

            new_matrix = fris_img_data[math.ceil(matrix_ns[5] - 10):math.ceil(matrix_ns[4] + 10), img_iw - 20:img_ie + 10]

            new_matrix_all = new_matrix_all + np.array(new_matrix)
        new_matrix_all = new_matrix_all / 4
        cv2.imwrite('dices_all/' + patient_allcase_folders[files_num][32:] + '_L3_L4.png', new_matrix_all)

        new_matrix_all = np.zeros((math.ceil(matrix_ns[6] + 10)-(math.ceil(matrix_ns[7] - 10)), (img_ie + 10 - img_iw + 20), 3))
        for img_num in range(1, 7):
            print('第'+str(img_num))
            patient_img = imgs_in_patient[img_num]
            fris_img_data = np.array(cv2.imread(patient_img, cv2.IMREAD_COLOR))

            new_matrix = fris_img_data[math.ceil(matrix_ns[7] - 10):math.ceil(matrix_ns[6] + 10), img_iw - 20:img_ie + 10]

            new_matrix_all = new_matrix_all + np.array(new_matrix)
        new_matrix_all = new_matrix_all / 6
        cv2.imwrite('dices_all/' + patient_allcase_folders[files_num][32:] + '_L2_L3.png', new_matrix_all)

        new_matrix_all = np.zeros((math.ceil(matrix_ns[8] + 10)-(math.ceil(matrix_ns[9] - 10)), (img_ie + 10 - img_iw + 20), 3))
        for img_num in range(1, 7):
            print('第'+str(img_num))
            patient_img = imgs_in_patient[img_num]
            fris_img_data = np.array(cv2.imread(patient_img, cv2.IMREAD_COLOR))

            new_matrix = fris_img_data[math.ceil(matrix_ns[9] - 10):math.ceil(matrix_ns[8] + 10), img_iw - 20:img_ie + 10]

            new_matrix_all = new_matrix_all + np.array(new_matrix)
        new_matrix_all = new_matrix_all / 6
        cv2.imwrite('dices_all/' + patient_allcase_folders[files_num][32:] + '_L1_L2.png', new_matrix_all)

        img_namelist = listdir('dices_all')  # 创建图像名列表
        for img_name in img_namelist:
            name = 'dices_all/' + img_name  # 创建文件名
            # 加载图片
            img = load_img(name, target_size=(224, 224))
            img = img_to_array(img)  # 转换为矩阵
            img = np.expand_dims(img, axis=0)  # 矩阵增维
            img = preprocess_input(img)  # 预处理：均值化

            feature = model.predict(img, verbose=0)  # 提取特征并保存
            features[img_name[:-4]] = feature  # 将向量写入字典
            print(img_name[:-4])
            print('>正在处理：', img_name)
        return features


if __name__ == '__main__':
    directory = 'G:\MRI_imagecaptain\png_t2_text'
    features = extract_features(directory)
    dump(features, open('features_cases.pkl', 'wb'))  # 保存文件
    print('图片特征提取完成，文件已保存！')
