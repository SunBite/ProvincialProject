# -*- coding: utf-8 -*-
from pyspark import SparkConf,SparkContext
import numpy as np
import GetFeatures
import os
import FeaturesExtractorOnSpark as feos
import cv2
import datetime

class FeaturesExtractorOnSpark:

    def __init__(self, keyframepath, featuressavepath,resizeheight=240, resizewidth=320):
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
        提取特征
        """
        sc = SparkContext(appName="FeaturesExtractorOnSpark"+os.path.basename(self.__featuressavepath))
        framenameandvaluerdd = sc.textFile(self.__keyframepath)

        def extractfeatures(line):
            """
            读取关键帧路径，进行特征提取
            :param line: hdfs上的一行数据，关键帧的路径
            :return: 关键帧的名字+“ ”+该关键帧的特征
            """
            framenamepath = line
            print(framenamepath)
            framebasenamelist = os.path.basename(framenamepath).split("_")

            framename = os.path.dirname(framenamepath) + os.sep + "_".join(framebasenamelist[:-1])
            framenum = float(framebasenamelist[-1].split(".")[0])
            frame = cv2.imread(framenamepath)
            # reshape
            frame = np.reshape(frame, (self.__resizeheight, self.__resizewidth, 3))
            # 获取关键帧的特征
            my_feature = GetFeatures.get_features(frame)

            # 获取特征list
            def getfeaturelist(my_feature):
                my_feature_temp = []
                for i in range(0, len(my_feature)):
                    for j in range(0, len(my_feature[i])):
                        my_feature_temp.append(float(my_feature[i][j]))
                return my_feature_temp

            my_feature = getfeaturelist(my_feature)
            print(len(my_feature))
            # 把帧编号加入到featurelist里
            my_feature.insert(0, framenum)

            return framename, my_feature
        def groupbyframename(line):
            """
            返回关键帧名字和相应的特征
            :param line: 关键帧的名字+“ ”+该关键帧的特征
            :return: myfeatureandname 返回关键帧名字和相应的特征
            """
            # 把帧名字加到list的第一个位置
            framename = line[0]
            ResultIterable = list(line[1])
            for i in range(0, len(ResultIterable)):
                if (ResultIterable[i][0] == float(0)):
                    my_feature_num1 = ResultIterable[i][1:]
                elif (ResultIterable[i][0] == float(1)):
                    my_feature_num2 = ResultIterable[i][1:]
                elif (ResultIterable[i][0] == float(2)):
                    my_feature_num3 = ResultIterable[i][1:]
            # 把3个关键帧的特征结合
            my_feature = my_feature_num1 + my_feature_num2 + my_feature_num3
            print(len(my_feature))
            my_feature.insert(0, framename)
            myfeatureandname = ""
            # 字符串拼接：帧的名字+“ ”+ 特征
            for i in range(0, len(my_feature)):
                myfeatureandname = myfeatureandname + " " + str(my_feature[i])
            # 返回关键帧名字和相应的特征
            return myfeatureandname
        # 分割和提取关键帧
        extractfeaturesfrombytesrdd = framenameandvaluerdd.map(extractfeatures).groupByKey().map(groupbyframename)
        # 保存到hdfs上
        extractfeaturesfrombytesrdd.saveAsTextFile(self.__featuressavepath)
        sc.stop()

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    feos.FeaturesExtractorOnSpark(
        "hdfs://sunbite-computer:9000/keyframepath.txt",
        "hdfs://sunbite-computer:9000/features320240-366-1/").featuresextractor()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('-------------FeaturesExtractorOnSpark Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')