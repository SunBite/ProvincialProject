# -*- coding: utf-8 -*-
import cv2
import os
import math
import numpy as np

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
        #保存关键帧列表
        self.__frames = []

    #设置要提取视频的路径
    # def setvideopath(self,videopath):
    #     if os.path.exists(videopath):
    #         self.__videopath = videopath
    #     else :
    #         print (" you inputted file is not existed!")

    def extractkeyframe(self):
        """
        提取关键帧
        """
        if os.path.exists(self.__videopath):
            videocapture = cv2.VideoCapture(self.__videopath)
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
                if self.__frames is not None:
                    for keyindex in range(len(self.__frames)):
                        currentframe = self.__frames[keyindex]
                        print(np.shape(currentframe))
                        if os.path.isdir(self.__savepath):
                            #存储关键帧的路径和名字
                            framename = os.path.abspath(self.__savepath) + os.sep + \
                                        os.path.basename(self.__videopath).split(".")[0] + \
                                        '_keyFrame_' + str(keyindex) + '.jpg'
                            cv2.imwrite(framename, currentframe)
                        else:
                            print(" Please input the correct save path!")
        else:
            print(" you inputted file is not existed!")