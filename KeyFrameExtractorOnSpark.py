# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
import KeyFrameExtractor
import datetime


class KeyFrameExtractorOnSpark:

    def __init__(self, videopath, savepath, resizeheight=240, resizewidth=320):
        """
        初始化方法
        :param videopath: 视频输入路径
        :param savepath: 关键帧保存路径
        :param resizeheight: 重置视频的大小的高
        :param resizewidth: 重置视频的大小的宽
        """
        self.__videopath = videopath
        self.__savepath = savepath
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth

    def keyframeextractor(self):
        """
        用spark关键帧提取
        :return:
        """
        sc = SparkContext(appName="KeyFrameExtractorOnSpark")
        videopathrdd = sc.textFile(self.__videopath)
        savepath = self.__savepath
        resizeheight = self.__resizeheight
        resizewidth = self.__resizewidth

        def extractkeyframe(line):
            """
            提取关键帧
            :param line: 要提取的视频路径
            :return:
            """
            for i in range(0, len(line) - 1):
                KeyFrameExtractor.KeyFrameExtractor(line[i], savepath, resizeheight, resizewidth).keyframeextractor()

        videopathrdd.map(lambda x: x.split(" ")).map(extractkeyframe).count()


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    KeyFrameExtractorOnSpark(r"file:/home/sunbite/videopath.txt", r"/home/sunbite/video/keyframe1/").keyframeextractor()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('-------------KeyFrameExtractorOnSpark Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
