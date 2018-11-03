# -*- coding: utf-8 -*-

import numpy as np
import myFirstPoint.GetFeatures as gf
import os
import cv2
import FeaturesExtractor as fe
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
        filepathandnamelist = filepathandname.split(os.sep)
        frame = cv2.imread(filepathandname)
        # reshape
        frame = np.reshape(frame, (self.__resizeheight, self.__resizewidth, 3))
        # 获取关键帧的特征
        # myFeature = gf.get_features(frame)
        # 获取不同组合的特征
        # _122feature = gf._122feature(myFeature)
        # _81feature = gf._81feature(myFeature)
        # _30_5_6feature = gf._30_5_6feature(myFeature)
        # _81_5_6feature = gf._81_5_6feature(myFeature)
        # _30feature = gf._30feature(myFeature)
        # _5_6feature = gf._5_6feature(myFeature)
        # _5feature = gf._5feature(myFeature)
        # _6feature = gf._6feature(myFeature)
        hogfeature = gf.hog_feature(frame)
        # orbfeature = gf.orb_feature(frame)
        # 要保存的不同组合的文件夹名字
        featureName = filename.replace("keyFrame", "feature").split(".")
        dirPath = os.sep + filepathandnamelist[-3] + os.sep + filepathandnamelist[-2] + os.sep
        _122featureDir = self.__featuressavepath + dirPath + "_122feature/"
        _81featureDir = self.__featuressavepath + dirPath + "_81feature/"
        _30_5_6featureDir = self.__featuressavepath + dirPath + "_30_5_6feature/"
        _81_5_6featureDir = self.__featuressavepath + dirPath + "_81_5_6feature/"
        _30featureDir = self.__featuressavepath + dirPath + "_30feature/"
        _5_6featureDir = self.__featuressavepath + dirPath + "_5_6feature/"
        _5featureDir = self.__featuressavepath + dirPath + "_5feature/"
        _6featureDir = self.__featuressavepath + dirPath + "_6feature/"
        hogfeatureDir = self.__featuressavepath + dirPath + "hogfeature/"
        orbfeatureDir = self.__featuressavepath + dirPath + "orbfeature/"

        # if not os.path.isdir(_122featureDir):
        #     os.makedirs(_122featureDir)
        if not os.path.isdir(_81featureDir):
            os.makedirs(_81featureDir)
        # if not os.path.isdir(_30_5_6featureDir):
        #     os.makedirs(_30_5_6featureDir)
        # if not os.path.isdir(_81_5_6featureDir):
        #     os.makedirs(_81_5_6featureDir)
        if not os.path.isdir(_30featureDir):
            os.makedirs(_30featureDir)
        # if not os.path.isdir(_5_6featureDir):
        #     os.makedirs(_5_6featureDir)
        # if not os.path.isdir(_5featureDir):
        #     os.makedirs(_5featureDir)
        # if not os.path.isdir(_6featureDir):
        #     os.makedirs(_6featureDir)
        if not os.path.isdir(hogfeatureDir):
            os.makedirs(hogfeatureDir)
        # if not os.path.isdir(orbfeatureDir):
        #     os.makedirs(orbfeatureDir)
        # 要保存的不同组合的特征
        # np.savetxt(_122featureDir + featureName[0], _122feature, newline=" ")
        # np.savetxt(_81featureDir + featureName[0], _81feature, newline=" ")
        # np.savetxt(_30_5_6featureDir + featureName[0], _30_5_6feature, newline=" ")
        # np.savetxt(_81_5_6featureDir + featureName[0], _81_5_6feature, newline=" ")
        # np.savetxt(_30featureDir + featureName[0], _30feature, newline=" ")
        # np.savetxt(_5_6featureDir + featureName[0], _5_6feature, newline=" ")
        # np.savetxt(_5featureDir + featureName[0], _5feature, newline=" ")
        # np.savetxt(_6featureDir + featureName[0], _6feature, newline=" ")
        np.savetxt(hogfeatureDir + featureName[0], hogfeature, newline=" ")
        # np.savetxt(orbfeatureDir + featureName[0], orbfeature, newline=" ")

    def getAllVideoFeature(self):
        """
        获取所有三帧组成的视频特征
        """
        classnames = ["basketball", "biking", "diving", "golf_swing", "horse_riding",
                      "soccer_juggling", "swing", "tennis_swing", "trampoline_jumping",
                      "volleyball_spiking", "walking"]
        testandtrains = ["test", "train"]
        # featureAndVideoFeatureDir = [("_122feature/", "_122videoFeature/"), ("_81feature/", "_81videoFeature/"),
        #                              ("_30_5_6feature/", "_30_5_6videoFeature/"),
        #                              ("_81_5_6feature/", "_81_5_6videoFeature/"),
        #                              ("_30feature/", "_30videoFeature/"),
        #                              ("_5_6feature/", "_5_6videoFeature/"),
        #                              ("_5feature/", "_5videoFeature/"),
        #                              ("_6feature/", "_6videoFeature/")
        #                              ]

        # featureAndVideoFeatureDir = [("_5_6feature/", "_5_6videoFeature/"),
        #                              ("_5feature/", "_5videoFeature/"),
        #                              ("_6feature/", "_6videoFeature/")
        #                              ]
        # featureAndVideoFeatureName = [("hogfeature", "hogvideoFeature")
        #                               ]
        featureAndVideoFeatureName = [("_81feature", "_81videoFeature")
                                      ]
        # featureAndVideoFeatureName = [("_30feature", "_30videoFeature")
        #                               ]
        # featureAndVideoFeatureDir = [("orbfeature/", "orbvideoFeature/")]
        for classname in classnames:
            for testandtrain in testandtrains:
                for featureName, videoFeatureName in featureAndVideoFeatureName:
                    featureDir = classname + os.sep + testandtrain + os.sep + featureName
                    videoFeatureDir = classname + os.sep + testandtrain + os.sep + videoFeatureName
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
            np.savetxt(videoFeatureDir + os.sep + os.path.basename(featureName), videoFeatureList, newline=" ")


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # fe.FeaturesExtractor(
    #     r"/home/sunbite/keyframe/",
    #     r"/home/sunbite/features_new_1").featuresExtractor()

    fe.FeaturesExtractor(
        r"/home/sunbite/MFSSEL/keyframe/",
        r"/home/sunbite/MFSSEL/features_new_1/").getAllVideoFeature()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '-------------FeaturesExtractor Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
