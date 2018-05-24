# coding: utf-8

import scipy.io
import skimage.io as skio
import skimage.color as skcolor
import numpy as np
import math
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops


"""
入口函数是 get_features(image_path)
"""


# 初始化全局变量
PI = math.pi
mapping = scipy.io.loadmat('mapping.mat')
mapping['table'] = np.array(mapping['mapping_table']).T
mapping['samples'] = int(mapping['mapping_samples'])
mapping['num'] = int(mapping['mapping_num'])

"""
HSV颜色特征提取
"""


def hsv_features(image_H, image_S, image_V, image_height, image_width):
    """
    获取图像的HSV颜色特征
    :param image_H: HSV图像的色调（Hue）分量
    :param image_S: HSV图像的饱和度（Saturation）分量
    :param image_V: HSV图像的亮度（Value）分量
    :param image_height: 图像的高度值
    :param image_width: 图像的宽度值
    :return: 处理得到的81位颜色特征向量
    """

    # 分区域计算HSV直方图
    t_feature = np.zeros(81)     # 初始化特征向量为0向量

    # 计算4x4区域中的第1列区域
    for i in range(math.floor(image_height/4), math.floor(image_height*3/4)):
        for j in range(math.floor(image_width/4)):
            tmp_h = math.floor(image_H[i][j]/0.111112)
            tmp_s = math.floor(image_S[i][j]/0.333334)
            tmp_v = math.floor(image_V[i][j]/0.333334)
            d = tmp_h * 9 + tmp_s * 3 + tmp_v + 1
            t_feature[d] = t_feature[d] + 1

    # 计算4x4区域中的第4列区域
    for i in range(math.floor(image_height/4), math.floor(image_height*3/4)):
        for j in range(math.floor(image_width*3/4), math.floor(image_width)):
            tmp_h = math.floor(image_H[i][j] / 0.111112)
            tmp_s = math.floor(image_S[i][j] / 0.333334)
            tmp_v = math.floor(image_V[i][j] / 0.333334)
            d = tmp_h * 9 + tmp_s * 3 + tmp_v + 1
            t_feature[d] = t_feature[d] + 1

    # 计算4x4区域中的第1行区域
    for i in range(math.floor(image_height/4)):
        for j in range(math.floor(image_width/4), math.floor(image_width*3/4)):
            tmp_h = math.floor(image_H[i][j] / 0.111112)
            tmp_s = math.floor(image_S[i][j] / 0.333334)
            tmp_v = math.floor(image_V[i][j] / 0.333334)
            d = tmp_h * 9 + tmp_s * 3 + tmp_v + 1
            t_feature[d] = t_feature[d] + 1

    # 计算4x4区域中的第4行区域
    for i in range(math.floor(image_height*3/4), math.floor(image_height)):
        for j in range(math.floor(image_width/4), math.floor(image_width*3/4)):
            tmp_h = math.floor(image_H[i][j] / 0.111112)
            tmp_s = math.floor(image_S[i][j] / 0.333334)
            tmp_v = math.floor(image_V[i][j] / 0.333334)
            d = tmp_h * 9 + tmp_s * 3 + tmp_v + 1
            t_feature[d] = t_feature[d] + 1

    # 计算4x4区域中的中间区域
    for i in range(math.floor(image_height * 3 / 4), math.floor(image_height)):
        for j in range(math.floor(image_width / 4), math.floor(image_width * 3 / 4)):
            tmp_h = math.floor(image_H[i][j] / 0.111112)
            tmp_s = math.floor(image_S[i][j] / 0.333334)
            tmp_v = math.floor(image_V[i][j] / 0.333334)
            d = tmp_h * 9 + tmp_s * 3 + tmp_v + 1
            t_feature[d] = t_feature[d] + 2

    return list(t_feature)


"""
LBP特征提取
"""


def apply_mapping(mat, mapping):
    """
    将LBP矩阵映射成特征向量
    :param mat: LBP矩阵
    :param mapping: 映射表
    :return: 映射后得到的特征向量
    """
    result = np.zeros([mat.shape[0], mat.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            result[i][j] = mapping['table'][int(mat[i][j])]
    single_result = []
    for i in result:
        single_result.extend(i)
    result_dict = {k: single_result.count(k)/(mat.shape[0]*mat.shape[1]) for k in set(single_result)}
    result = [result_dict[k] for k in result_dict]
    return result


def get_lbp(r, n, image_gray):
    """
    使用LBP算子提取灰度图像的LBP特征图，并将其映射到特征向量
    :param r: lbp算子半径
    :param n: lbp算子选取的邻接点数
    :param image_gray: 待处理的灰度图像
    :return: 10维特征向量
    """
    # lbp算子初始化
    radius = r
    n_points = n

    lbp = local_binary_pattern(image_gray, n_points, radius)  # 提取LBP特征
    lbp_vec = apply_mapping(lbp, mapping)                # 映射特征到特征向量
    return lbp_vec


def lbp_feature(image_gray):
    return get_lbp(2, 8, image_gray)+ get_lbp(3, 8, image_gray) + get_lbp(4, 8, image_gray)

"""
Tchebichef矩特征提取
不要试图去看懂这部分代码！！！！！！
将这段代码从前辈的matlab代码改写到python的时候我已经尝试过这件事情并且感受到了什么叫绝望！！！！
"""


def F(r, v, image_height, image_width, image):
    x = int(r*math.cos(PI*v/180)+image_height/2)
    y = int(r*math.sin(PI*v/180)+image_width/2)
    f = float(image[x][y])/255
    return f


def B(v, q, r, image_height, image_width, image):
    b1 = F(r, 0, image_height, image_width, image)
    b2 = F(r, 1, image_height, image_width, image)

    for i in range(2,359):
        b1 = F(r,v,image_height, image_width, image)+b2*2*math.cos(q)-b1
        temp = b1
        b1 = b2
        b2 = temp
    return b1, b2


def T(p, r, N):
    if p == 0:
        t = 1
    elif p == 1:
        t = (2*r+1-N)/N
    else:
        t = ((2*p-1)*(2*r+1-N)/N*T(p-1, r, N)-(p-1)*T(p-2,r,N)*(N**p-(p-1)**2))/p/N**p
    return t


def fast_tchebichef(p, q, image_height, image_width, image):
    """
    化简后的Tchebichef矩计算
    :param p:
    :param q:
    :param image_height: 灰度图像的高
    :param image_width: 灰度图像的宽
    :param image: 灰度图像
    :return:
    """
    m = int(min(image_height/2-1, image_width/2-1))
    sum = 0
    I = math.cos(q*358*PI/180)
    K = math.cos(q*359*PI/180)
    for i in range(m):
        b1, b2 = B(358, q, i, image_height, image_width, image)
        sum = sum+(b2*I+F(i, 359, image_height, image_width, image)*K
                   - F(i, 0, image_height, image_width, image)-b1*K)*T(p, i, image_width)
    S = sum/360
    return S


def tchebichef_features(image_gray, image_height, image_width):
    """
    利用化简后的Tchebichef矩计算函数，计算出5位径向Tchebichef矩特征向量
    :param image_gray: 待提取特征的灰度图像
    :param image_height: 图像的高度
    :param image_width: 图像的宽度
    :return: 5维Tchebichef矩特征向量
    """
    t_feature = np.zeros(5)
    for i in range(5):
        t_feature[i] = (fast_tchebichef(i, 8, image_height, image_width, image_gray) + 1) / 2
    return list(t_feature)


"""
纹理特征提取
"""


def glcm_feature(image_gray):
    """
    基于灰度共生矩阵的纹理特征提取
    :param image_gray: 灰度图像
    :return: 灰度图像的六种纹理特征
    """
    image_gray = np.uint(np.floor(image_gray / 0.33333334))
    image_glcm = greycomatrix(image_gray, [1], [0], 3)
    image_glcm_feature = [int(greycoprops(image_glcm, 'contrast')),
                          int(greycoprops(image_glcm, 'dissimilarity')),
                          int(greycoprops(image_glcm, 'homogeneity')),
                          int(greycoprops(image_glcm, 'energy')),
                          int(greycoprops(image_glcm, 'ASM')),
                          int(greycoprops(image_glcm, 'correlation'))]
    return image_glcm_feature


"""
特征计算与融合，分别提取四种特征，并将四种特种融合成一个122维的特征向量
"""


def get_features(image):
    # 图像预处理
    #image = skio.imread(image_path)  # 载入图像
    image_gray = skcolor.rgb2gray(image)       # 原始图像转换为灰度图像
    image_hsv = skcolor.rgb2hsv(image)    # 原始图像转换为HSV图像
    image_hsv_H = np.array(image_hsv[:, :, 0]) / 255  # HSV色调（Hue）分量
    image_hsv_S = np.array(image_hsv[:, :, 1]) / 255  # HSV饱和度(Saturation)分量
    image_hsv_V = np.array(image_hsv[:, :, 2]) / 255  # HSV亮度（Value）分量
    image_height = image.shape[0]         # 图像的高度值
    image_width = image.shape[1]          # 图像的宽度值

    # 颜色特征提取
    return hsv_features(image_hsv_H, image_hsv_S, image_hsv_V, image_height, image_width) + \
           glcm_feature(image_gray) + \
           tchebichef_features(image_gray, image_height, image_width) + \
           lbp_feature(image_gray)


# 测试脚本
if __name__ == '__main__':
    my_feature = get_features('v_shooting_01_01_0.jpg')
    print(my_feature)
    print(len(my_feature))
    pass