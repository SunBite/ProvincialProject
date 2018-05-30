# -*- coding: utf-8 -*-
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
import time

class VideoSparkStreamingConsumer:

    def __init__(self, savepath, topicname,brokerlist,resizeheight=240, resizewidth=320):
        """
        初始化方法
        :param savepath: 关键帧保存路径
        :param topicname: topic名字
        :param brokerlist: brokerlist
        :param resizeheight: 重置视频的大小的高
        :param resizewidth: 重置视频的大小的宽
        """
        self.__savepath = savepath
        self.__topicname = [topicname]
        self.__brokerlist = brokerlist
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth



    def savekeyframe(self):
        """
        保存关键帧
        """
        sc = SparkContext(appName="VideoSparkStreamingConsumer")
        ssc = StreamingContext(sc, 2)
        messages = KafkaUtils.createDirectStream(ssc=ssc, topics=self.__topicname,
                                                 kafkaParams={"metadata.broker.list":self.__brokerlist})
        # 把frame的（framename,frametobytes）存到savepath中
        def framehandler(rdd):
            if not (rdd.isEmpty()):

                def bytestounit8codechanger(frame):
                    """
                    把dtype=bytes的flatten ndarray转换成dtype=str的list
                    :param frame: dtype=bytes的flatten ndarray
                    :return: frame: dtype=str的list
                    """
                    # 把ndarray.bytes转成ndarray.str
                    bytestostrdecoder = np.vectorize(lambda x: bytes.decode(x))
                    frame = bytestostrdecoder(frame)
                    # 把ndarray.str转成ndarray.uint8
                    #strtouint8decoder = np.vectorize(lambda x: np.uint8(x))
                    #frame = strtouint8decoder(frame)
                    return list(frame)

                def getframenameandlist(line):
                    """
                    从producer上读取数据
                    :param line: producer传过来的数据，（关键帧的名字，关键帧的字节码）
                    :return: 关键帧的名字+“ ”+该关键帧的特征list
                    """
                    framename = str(line[0])
                    print(framename)
                    # 把bytes转ndarray
                    frame = np.fromstring(line[1], dtype="S3")
                    # 把dtype=bytes的flatten ndarray转换成dtype=str的list
                    frame = bytestounit8codechanger(frame)
                    # 把帧名字加到list的第一个位置
                    frame.insert(0, framename)
                    nameandframelist = ""
                    # 字符串拼接：帧的名字+“ ”+ 特征
                    for i in range(0, len(frame)):
                        nameandframelist = nameandframelist + " " + frame[i]
                    return nameandframelist

                rdd.map(getframenameandlist).saveAsTextFile(self.__savepath + str(int(round(time.time()*1000))))

        messages.foreachRDD(framehandler)
        #messages.saveAsTextFiles(self.__savepath)
        ssc.start()
        ssc.awaitTermination()


if __name__ == '__main__':
    vssc = VideoSparkStreamingConsumer(savepath="hdfs://sunbite-computer:9000/keyframe320240-366/keyframe-walking/keyframe-walking", topicname=r"test", brokerlist="10.3.11.131:9092")
    #vssc = VideoSparkStreamingConsumer(savepath="file:/home/sunbite/keyframe320240/keyframe-walking/keyframe-walking", topicname=r"test", brokerlist="10.3.11.131:9092")
    vssc.savekeyframe()
