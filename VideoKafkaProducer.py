# -*- coding: utf-8 -*-
import threading
import cv2
import os
from kafka import KafkaProducer
import math
import numpy as np

class VideoKafkaProducer(threading.Thread):

    def __init__(self, videopath, brokerurl, topic, resizeheight=240, resizewidth=320):
        """
            初始化方法
            :param videopath: 视频输入路径
            :param brokerurl  broker路径
            :param topic topic名称
            :param resizeheight: 重置视频的大小的高
            :param resizewidth: 重置视频的大小的宽
            """
        threading.Thread.__init__(self)
        self.__videopath = videopath
        self.__brokerurl = brokerurl
        self.__topic = topic
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth
        self.__producer = KafkaProducer(bootstrap_servers=brokerurl)
        # 保存关键帧列表
        self.__frames = []

    def run(self):
        """
        把关键帧放到通过kafka传送出去
        """
        if os.path.isdir(self.__videopath):
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(self.__videopath):
                for filename in filenames:
                    filepathandname = os.path.join(dirpath, filename)
                    self.__extractkeyframe(filepathandname)
                    self.__producersend(filepathandname)

    def __extractkeyframe(self, videopath):
        """
        提取关键帧
        """
        if os.path.exists(videopath):
            videocapture = cv2.VideoCapture(videopath)
            if videocapture.isOpened():
                #所有帧
                wholeframenum = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
                if wholeframenum < 2:
                    print('the image you inputted has not enougth frames!')
                #中间帧
                middleframenum = math.ceil(wholeframenum/2)
                #获取第一帧
                success, frame = videocapture.read()
                frame = cv2.resize(frame, (self.__resizeheight, self.__resizewidth))
                #加入第一帧
                self.__frames = [frame]
                count = 0
                while success:
                    count += 1
                    #获取下一帧
                    success, frame = videocapture.read()
                    if success:
                        frame = cv2.resize(frame, (self.__resizeheight, self.__resizewidth))
                        #加入中间帧
                        if count == middleframenum:
                            self.__frames.append(frame)
                        #加入最后一帧
                        elif count == wholeframenum - 1:
                            self.__frames.append(frame)

        else:
            print(" you inputted file is not existed!")

    def __producersend(self, file):
        """
        producer发送
        :param file: 文件路径
        :return:
        """
        if self.__frames is not None:
            for keyindex in range(len(self.__frames)):
                currentframe = self.__frames[keyindex]
                # 存储关键帧的路径和名字
                framename = os.path.dirname(file) + os.sep + \
                            os.path.split(os.path.dirname(file))[1] + '_' + \
                            os.path.basename(file).split(".")[0] + \
                            '_keyFrame_' + str(keyindex)
                # 把dtype=unit8的ndarray转换成dtype=bytes的flatten ndarray
                currentframe = self.__unit8tobytescodechanger(currentframe)
                #把ndarray转换成bytes
                frametobytes = np.ndarray.tobytes(currentframe)
                producer = self.__producer
                producer.send(topic=self.__topic, value=frametobytes,
                                  key=str.encode(framename))
                producer.flush()
        else:
            print(" you inputted file is not existed!")

    def __unit8tobytescodechanger(self, frame):
        """
        把dtype=unit8的ndarray转换成dtype=bytes的flatten ndarray
        :param frame: dtype=unit8的ndarray
        :return: frame: dtype=bytes的flatten ndarray
        """
        # 把ndarray.uint8转换成把ndarray.str
        uint8tostrdecoder = np.vectorize(lambda x: str(x))
        frame = uint8tostrdecoder(frame)
        # 把ndarray平整化
        frame = np.ndarray.flatten(frame)
        # 把ndarray.str转成ndarray.bytes
        strtobytesdecoder = np.vectorize(lambda x: str.encode(x))
        frame = strtobytesdecoder(frame)
        return frame

if __name__ == '__main__':
    kafkaproducer = VideoKafkaProducer(r"/home/sunbite/video/action_youtube_naudio/walking", r"10.3.11.131:9092", r"test")
    kafkaproducer.start()
