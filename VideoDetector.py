# -*- coding: utf-8 -*-
import cv2
import os
import math
import GetFeatures
from sklearn.externals import joblib


class VideoDetector:
    def __init__(self, filepath, modelpath, resizeheight=240, resizewidth=320):
        # 保存关键帧列表
        self.__frames = []
        self.__features = []
        self.__filepath = filepath
        self.__modelpath = modelpath
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth
        self.__classmap = {"basketball": 1, "biking": 2, "diving": 3, "golf_swing": 4, "horse_riding": 5,
                           "soccer_juggling": 6, "swing": 7, "tennis_swing": 8, "trampoline_jumping": 9,
                           "volleyball_spiking": 10,
                           "walking": 11}

    def extractkeyframe(self):
        """
        提取关键帧
        """
        if os.path.exists(self.__filepath):
            videocapture = cv2.VideoCapture(self.__filepath)
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
                self.__frames.insert(0, self.__filepath)
        else:
            print(" you inputted file is not existed!")

    def getFeatures(self):
        """
        获得三帧的对应的特征
        :return:
        """
        self.extractkeyframe()
        if len(self.__frames) == 4:
            features = []
            for i in range(0, len(self.__frames) - 1):
                features.extend(GetFeatures.get_features(self.__frames[i + 1]))
            self.__features = self.getFeaturesList(features)
            classname = os.path.basename(os.path.dirname(self.__frames[0]))
            classnum = self.__classmap[classname]
            # 把真正的标签放在特征list的第一个位置
            self.__features.insert(0, float(classnum))
        else:
            print("your keyframes are not enough!!")
        return self.__features

    def getLabel(self):
        """
        获取label
        :return:
        """
        self.getFeatures()
        if os.path.exists(self.__modelpath):
            # 加载model文件
            model = joblib.load(self.__modelpath)
            test_y = [self.__features[0]]
            test_x = [self.__features[1:]]
            # 预测
            predict_label = model.predict(test_x)
            return predict_label[0], test_y[0]

    def getFeaturesList(self, my_feature):
        """
        获取特征list
        :return:
        """
        my_feature_temp = []
        for i in range(0, len(my_feature)):
            for j in range(0, len(my_feature[i])):
                my_feature_temp.append(float(my_feature[i][j]))
        return my_feature_temp
