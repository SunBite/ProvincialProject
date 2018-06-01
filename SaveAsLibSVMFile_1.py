# -*- coding: utf-8 -*-

from pyspark import SparkContext
import os
from ListParam import *
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import SaveAsLibSVMFile as salf

class SaveAsLibSVMFile:
    def __init__(self, featurespath, savepath):
        """
        初始化方法
        :param featurespath: 输入的特征路径
        :param savepath: 保存的libSVMFile路径
        """
        self.__featurespath = featurespath
        self.__savepath = savepath
        self.__classmap = {"basketball": 1, "biking": 2, "diving": 3, "golf_swing": 4, "horse_riding": 5, "soccer_juggling": 6,

                           "swing": 7, "tennis_swing": 8, "trampoline_jumping": 9, "volleyball_spiking": 10, "walking": 11}


    def saveaslibSVMfile(self):
        """
        保存libsvm格式的文件
        :return:
        """
        sc = SparkContext(master="local[2]",appName="SaveAsLibSVMFile"+os.path.basename(self.__savepath))
        features = sc.textFile(self.__featurespath)
        TOTALFEATUREANDLABEL = sc.accumulator([], ListParamForLabeledPoint())

        def codechange(line):
            """
            根据“_v”切分出类别信息
            :param line:关键帧的特征
            :return: （类别号，特征）
            """
            classname = os.path.basename(line[0]).split("_v")[0]
            classnum = self.__classmap[classname]
            # ResultIterable = list(line[1])
            # features = ResultIterable[0] + ResultIterable[1] + ResultIterable[2]
            # print(len(features))
            return (classnum, list(line[1]))

        def getfeaturesandlabel(line):
            """
            返回LabeledPoint类型的标签和特征组合
            :param line:（类别号，特征）
            :return:返回LabeledPoint类型的标签和特征组合
            """
            #global TOTALFEATUREANDLABEL
            return LabeledPoint(line[0], Vectors.dense(line[1]))
            #TOTALFEATUREANDLABEL += [LabeledPoint(line[0], Vectors.dense(line[1]))]

        featuresandlabel = features.map(lambda x: x.split(" ")).map(lambda x: (x[1], x[2:])).map(codechange).map(
                getfeaturesandlabel).repartition(1)
        featuresandlabel.count()
        print(featuresandlabel.count())
        #totalfeatureandlabel = TOTALFEATUREANDLABEL.value
        MLUtils.saveAsLibSVMFile(featuresandlabel, self.__savepath)
        sc.stop()
if __name__ == '__main__':
    classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling", "swing",
                 "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
    for i in classname:
        print(i)
        #salf.SaveAsLibSVMFile("hdfs://sunbite-computer:9000/features320240-366/features-" + i,
        #                      "/home/sunbite/libsvmfile320240-366/" + i).saveaslibSVMfile()
        salf.SaveAsLibSVMFile("file:/home/sunbite/features320240-366/features-" + i,
                              "file:/home/sunbite/libsvmfile320240-366/" + i).saveaslibSVMfile()
