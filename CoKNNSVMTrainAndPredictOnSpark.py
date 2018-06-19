# -*- coding: utf-8 -*-
import svmutil
from pyspark import SparkContext
import CoKNNSVMTrainAndPredictOnSpark as ckstapos
import Co_KNN_SVM
import Co_KNN_SVM_New
import Co_KNN_SVM_Utilities
import random
import datetime
import os
from ListParam import *
import test1
from sklearn.model_selection import train_test_split


class CoKNNSVMTrainAndPredictOnSpark:

    def __init__(self, filepath, savepath):
        """
        初始化方法
        :param filepath: hdfs上的要读取的features目录
        :param savepath: 保存的地址
        """
        self.__filepath = filepath
        self.__savepath = savepath
        self.__classmap = {"basketball": 1, "biking": 2, "diving": 3, "golf_swing": 4, "horse_riding": 5,
                           "soccer_juggling": 6, "swing": 7, "tennis_swing": 8, "trampoline_jumping": 9,
                           "volleyball_spiking": 10,
                           "walking": 11}

    def CoKNNSVMTrainAndPredictOnSpark(self):
        """
        训练模型，预测结果
        """
        global TOTALFEATURESANDLABEL
        sc = SparkContext(appName="CoKNNSVMTrainAndPredictOnSpark")
        TOTALFEATURESANDLABEL = sc.accumulator([], ListParamForFeatureAndLabel())
        features = sc.textFile(self.__filepath)

        def makefeatures(line):
            """
            根据“_v”切分出类别信息
            :param line:关键帧的特征
            """
            classname = os.path.basename(line[0]).split("_v")[0]
            classnum = self.__classmap[classname]
            return (float(classnum), [float(x) for x in line[1]])

        def getmodelandaccuary(line):
            """
            训练模型，预测结果
            :param line: hdfs上的要读取的features目录的目录
            :return: 准确率
            """
            global TOTALFEATURESANDLABEL
            TOTALFEATURESANDLABEL += [(line[0], line[1])]

        # features.map(lambda x:x.split(" ")).map(getmodelandaccuary).repartition(1).saveAsTextFile(self.__savepath)
        features.map(lambda x: x.split(" ")).map(lambda x: (x[1], x[2:])).map(makefeatures).map(
            getmodelandaccuary).count()
        totalfeaturesandlabel = TOTALFEATURESANDLABEL.value

        def getfeaturelistandlabellist(totalfeaturesandlabel):
            """
            把累加器中的label和特征的元组提出来，形成标签list和featrueslist
            :param totalfeaturesandlabel:label和特征的元组
            :return:（标签list，featrueslist）
            """
            TOTALFEATURES = []
            TOTALLABEL = []
            for i in range(0, len(totalfeaturesandlabel)):
                TOTALLABEL.append(totalfeaturesandlabel[i][0])
                TOTALFEATURES.append(totalfeaturesandlabel[i][1])
            return (TOTALLABEL, TOTALFEATURES)

        totallabel, totalfeatures = getfeaturelistandlabellist(totalfeaturesandlabel)
        # y = totallabel
        # x = totalfeatures

        # x = Co_KNN_SVM_Utilities.getfeatureforlibsvm(x)

        random_index = [i for i in range(len(totallabel))]
        #test_random_index = [i for i in range(len(x))]
        random.shuffle(random_index)
        #random.shuffle(test_random_index)
        random_y = [totallabel[x] for x in random_index]
        random_x = [totalfeatures[x] for x in random_index]
        # random_test_y = [test_y[x] for x in test_random_index]
        # random_test_x = [test_x[x] for x in test_random_index]
        # random_train_y = [train_y[x] for x in train_random_index]
        # random_train_x = [train_x[x] for x in train_random_index]
        # random_test_y = [test_y[x] for x in test_random_index]
        # random_test_x = [test_x[x] for x in test_random_index]
        # random_train_y = train_y
        # random_train_x = train_x
        # random_test_y = test_y
        # random_test_x = test_x
        # train_y = random_y[0:1500]
        # train_x = random_x[0:1500]
        # test_y = random_y[1500:1580]
        # test_x = random_x[1500:1580]
        train_x,test_x,train_y,test_y=train_test_split(totalfeatures,totallabel,test_size=0.2,random_state=42,
                                                                                          shuffle=False)
        # train_y = totallabel[0:800]
        # train_x = totalfeatures[0:800]
        # test_y = totallabel[800:1580]
        # test_x = totalfeatures[800:1580]
        #Co_KNN_SVM.Co_KNN_SVM(train_y, train_x, test_y, test_x, self.__savepath)
        #test1.Co_KNN_SVM(train_y, train_x, test_y, test_x, self.__savepath)
        Co_KNN_SVM_New.Co_KNN_SVM(train_y, train_x, test_y, test_x, self.__savepath)


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # ckstapos.CoKNNSVMTrainAndPredictOnSpark("hdfs://sunbite-computer:9000/filepath/filepath320240-366.txt",
    ckstapos.CoKNNSVMTrainAndPredictOnSpark("file:/home/sunbite/Co_KNN_SVM_TMP/features320240-366/",
                                            '/home/sunbite/Co_KNN_SVM_TMP/CoKNNSVM1.model').CoKNNSVMTrainAndPredictOnSpark()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '----------CoKNNSVMTrainAndPredictOnSpark Running time: %s Seconds-----------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
