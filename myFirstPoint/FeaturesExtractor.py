# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import myFirstPoint.GetFeatures as gf
import os
import cv2
#import FeaturesExtractor as fe
import datetime


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

    def featuresExtractor(self):
        """
        遍历文件夹下所有的关键帧，进行特征提取
        :return:
        """
        if os.path.isdir(self.__keyframepath):
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(self.__keyframepath):
                for filename in filenames:
                    # filepathandname = os.path.join(dirpath, filename)
                    self.__extractFeatures(dirpath, filename)

    def __extractFeatures(self, dirpath, filename):
        """
        提取特征
        """
        filepathandname = os.path.join(dirpath, filename)
        frame = cv2.imread(filepathandname)
        # reshape
        frame = np.reshape(frame, (self.__resizeheight, self.__resizewidth, 3))
        # 获取关键帧的特征
        myFeature = gf.get_features(frame)
        # 获取不同组合的特征
        _122feature = gf._122feature(myFeature)
        _81feature = gf._81feature(myFeature)
        _30_5_6feature = gf._30_5_6feature(myFeature)
        _81_5_6feature = gf._81_5_6feature(myFeature)
        _30feature = gf._30feature(myFeature)
        # 要保存的不同组合的文件夹名字
        featureName = filename.replace("keyFrame", "feature").split(".")
        _122featureDir = self.__featuressavepath + "_122feature/"
        _81featureDir = self.__featuressavepath + "_81feature/"
        _30_5_6featureDir = self.__featuressavepath + "_30_5_6feature/"
        _81_5_6featureDir = self.__featuressavepath + "_81_5_6feature/"
        _30featureDir = self.__featuressavepath + "_30feature/"

        if not os.path.isdir(_122featureDir):
            os.makedirs(_122featureDir)
        if not os.path.isdir(_81featureDir):
            os.makedirs(_81featureDir)
        if not os.path.isdir(_30_5_6featureDir):
            os.makedirs(_30_5_6featureDir)
        if not os.path.isdir(_81_5_6featureDir):
            os.makedirs(_81_5_6featureDir)
        if not os.path.isdir(_30featureDir):
            os.makedirs(_30featureDir)
        # 要保存的不同组合的特征
        np.savetxt(_122featureDir + featureName[0], _122feature, newline=" ")
        np.savetxt(_81featureDir + featureName[0], _81feature, newline=" ")
        np.savetxt(_30_5_6featureDir + featureName[0], _30_5_6feature, newline=" ")
        np.savetxt(_81_5_6featureDir + featureName[0], _81_5_6feature, newline=" ")
        np.savetxt(_30featureDir + featureName[0], _30feature, newline=" ")

    def getAllVideoFeature(self):
        """
        获取所有三帧组成的视频特征
        """
        featureAndVideoFeatureDir = [("_122feature/", "_122videoFeature/"), ("_81feature/", "_81videoFeature/"),
                                     ("_30_5_6feature/", "_30_5_6videoFeature/"),
                                     ("_81_5_6feature/", "_81_5_6videoFeature/"),
                                     ("_30feature/", "_30videoFeature/")]
        for featureDir, videoFeatureDir in featureAndVideoFeatureDir:
            self.__getVideoFeature(featureDir, videoFeatureDir)

    def __getVideoFeature(self, featureDir, videoFeatureDir):
        """
        获取三帧组成的视频特征
        """
        featureNameList = []
        featureDir = self.__featuressavepath + featureDir
        videoFeatureDir = self.__featuressavepath + videoFeatureDir
        if os.path.isdir(featureDir):
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(featureDir):
                for filename in filenames:
                    filename = filename[:-10]
                    filepathandname = os.path.join(dirpath, filename)
                    featureNameList.append(filepathandname)
        featureNameList.sort()
        # 去重
        featureNameListNew = list(set(featureNameList))

        if not os.path.isdir(videoFeatureDir):
            os.makedirs(videoFeatureDir)
        for featureName in featureNameListNew:
            videoFeatureList = []
            for i in range(0, 3):
                videoFeatureList.extend(np.loadtxt(featureName + "_feature_" + str(i)))
            np.savetxt(videoFeatureDir + os.path.basename(featureName), videoFeatureList, newline=" ")


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # fe.FeaturesExtractor(
    #     r"/home/sunbite/Co_KNN_SVM_TMP/keyframe/",
    #     r"/home/sunbite/features/").featuresExtractor()

    fe.FeaturesExtractor(
        r"/home/sunbite/Co_KNN_SVM_TMP/keyframe/",
        r"/home/sunbite/features/").getAllVideoFeature()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '-------------FeaturesExtractor Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
