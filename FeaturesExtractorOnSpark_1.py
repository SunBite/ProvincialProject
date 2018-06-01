# -*- coding: utf-8 -*-
from pyspark import SparkConf,SparkContext
import numpy as np
import GetFeatures
import os
import FeaturesExtractorOnSpark as feos

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

    def __bytestounit8codechanger(self, frame):
        """
        把dtype=bytes的flatten ndarray转换成dtype=unit8的ndarray
        :param frame: dtype=bytes的flatten ndarray
        :return: frame: dtype=unit8的ndarray
        """
        # 把ndarray.bytes转成ndarray.str
        bytestostrdecoder = np.vectorize(lambda x: bytes.decode(x))
        frame = bytestostrdecoder(frame)
        # 把ndarray.str转成ndarray.uint8
        strtouint8decoder = np.vectorize(lambda x: np.uint8(x))
        frame = strtouint8decoder(frame)
        return frame

    def extractfeatures(self):
        """
        提取特征
        """
        sc = SparkContext(master="local[2]",
                          appName="FeaturesExtractorOnSpark"+os.path.basename(self.__featuressavepath))
        framenameandvaluerdd = sc.textFile(self.__keyframepath)

        def extractfeaturesfrombytes(line):
            """
            从hdfs上读取的数据进行编码转换，得到关键帧的名字和相应的帧list，
            然后进行还原，然后进行特征提取
            :param line: hdfs上的一行数据，包括关键帧的名字和关键帧de 帧list
            :return: 关键帧的名字+“ ”+该关键帧的特征
            """
            framename = line[0]
            print(framename)
            # 把ndarray.str转成ndarray.uint8
            strtouint8decoder = np.vectorize(lambda x: np.uint8(x))
            frame = strtouint8decoder(line[1])
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
            # 把帧名字加到list的第一个位置
            my_feature.insert(0,framename)
            myfeatureandname = ""
            # 字符串拼接：帧的名字+“ ”+ 特征
            for i in range(0,len(my_feature)):
                myfeatureandname = myfeatureandname + " " + str(my_feature[i])
            # 返回关键帧名字和相应的特征
            return myfeatureandname
        # 分割和提取关键帧
        extractfeaturesfrombytesrdd = framenameandvaluerdd.map(lambda x: x.split(" ")).map(lambda x:(x[1],x[2:])).map(extractfeaturesfrombytes)
        # 保存到hdfs上
        extractfeaturesfrombytesrdd.saveAsTextFile(self.__featuressavepath)
        sc.stop()

if __name__ == '__main__':
    classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling", "swing",
                 "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
    for i in classname:
        feos.FeaturesExtractorOnSpark(
            "hdfs://sunbite-computer:9000/keyframe320240/keyframe-" + i + "/keyframe-" + i + "*/part-*",
            "file:/home/sunbite/features320240-122/features-" + i + "/").extractfeatures()