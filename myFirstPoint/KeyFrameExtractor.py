# -*- coding: utf-8 -*-
import cv2
import os
import math
import numpy as np
import datetime


class KeyFrameExtractor:

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
        # 保存关键帧列表
        self.__frames = []

    def keyframeextractor(self):
        """
        遍历文件夹下所有的视频，进行关键帧提取
        :return:
        """
        if os.path.isdir(self.__videopath):
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(self.__videopath):
                for filename in filenames:
                    filepathandname = os.path.join(dirpath, filename)
                    self.__extractkeyframe(filepathandname)

    def __extractkeyframe(self, filepathandname):
        """
        提取关键帧
        """
        if os.path.exists(filepathandname):
            videocapture = cv2.VideoCapture(filepathandname)
            if videocapture.isOpened():
                # 所有帧
                wholeframenum = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
                if wholeframenum < 2:
                    print('the image you inputted has not enougth frames!')
                # 中间帧
                middleframenum = math.ceil(wholeframenum / 2)
                # 获取第一帧
                success, frame = videocapture.read()
                frame = cv2.resize(frame, (self.__resizewidth, self.__resizeheight))
                # 加入第一帧filepathandname
                self.__frames = [frame]
                count = 0
                while success:
                    count += 1
                    # 获取下一帧
                    success, frame = videocapture.read()
                    if success:
                        frame = cv2.resize(frame, (self.__resizewidth, self.__resizeheight))
                        # 加入中间帧
                        if count == middleframenum:
                            self.__frames.append(frame)
                        # 加入最后一帧
                        elif count == wholeframenum - 1:
                            self.__frames.append(frame)
                if self.__frames is not None:
                    for keyindex in range(len(self.__frames)):
                        currentframe = self.__frames[keyindex]
                        if os.path.isdir(self.__savepath):
                            # 存储关键帧的路径和名字
                            framename = os.path.abspath(self.__savepath) + os.sep + \
                                        os.path.basename(os.path.dirname(filepathandname)) + "_" + \
                                        os.path.basename(filepathandname).split(".")[0] + \
                                        '_keyFrame_' + str(keyindex) + '.jpg'
                            cv2.imwrite(framename, currentframe)
                        else:
                            print(" Please input the correct save path!")
        else:
            print(" you inputted file is not existed!")
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    KeyFrameExtractor = KeyFrameExtractor(r"/home/sunbite/video/action_youtube_naudio", r"/home/sunbite/video/keyframe2")
    KeyFrameExtractor.keyframeextractor()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('-----------------KeyFrameExtractor Running time: %s Seconds-----------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')