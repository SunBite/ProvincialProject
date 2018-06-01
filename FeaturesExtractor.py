# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import GetFeatures
import os
import cv2
class FeaturesExtractor:
    def __init__(self, keyframepath, featuressavepath, resizeheight=240, resizewidth=320):
        """
        初始化方法
        :param keyframepath: hdfs上的keyframe信息的路径
        :param featuressavepath: hdfs上的features保存的路径
        :param resizeheight: 重置视频的大小的高
        :param resizewidth: 重置视频的大小的宽
        """
        self.__keyframepath = keyframepath
        self.__featuressavepath = featuressavepath
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth

    def featuresextractor(self):
        """
        遍历文件夹下所有的关键帧，进行特征提取
        :return:
        """
        if os.path.isdir(self.__keyframepath):
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(self.__keyframepath):
                for filename in filenames:
                    filepathandname = os.path.join(dirpath, filename)
                    self.__extractfeatures(filepathandname)

    def __extractfeatures(self, filepathandname):
        """
        提取特征
        """
        frame = cv2.imread(filepathandname)
        # reshape
        frame = np.reshape(frame, (self.__resizeheight, self.__resizewidth, 3))
        # 获取关键帧的特征
        my_feature = GetFeatures.get_features(frame)