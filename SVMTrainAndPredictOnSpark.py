# -*- coding: utf-8 -*-
import svmutil
from pyspark import SparkContext
import SVMTrainAndPredictOnSpark as stapos

class SVMTrainAndPredictOnSpark:
    def __init__(self, filepath, savepath):
        """
        初始化方法
        :param filepath: hdfs上的要读取的features目录的目录
        :param savepath: 保存的地址
        """
        self.__filepath = filepath
        self.__savepath = savepath

    def SVMTrainAndPredictOnSpark(self):
        """
        训练模型，预测结果
        """
        sc = SparkContext(master="local[2]", appName="SVMTrainAndPredictOnSpark")
        features = sc.textFile(self.__filepath)
        def getmodelandaccuary(line):
            """
            训练模型，预测结果
            :param line: hdfs上的要读取的features目录的目录
            :return: 准确率
            """
            train_y = []
            train_x = []
            test_y = []
            test_x = []
            for i in range(0,len(line)-1):
                y, x = svmutil.svm_read_problem(line[i])
                train_y.extend(y[0:240])
                train_x.extend(x[0:240])
                test_y.extend(y[200:300])
                test_x.extend(x[200:300])
            print(train_x[0])
            m = svmutil.svm_train(train_y, train_x, "-s 0 -t 2 -c 32 -g 8 -b 1")
            predict_label, accuary, prob_estimates = svmutil.svm_predict(test_y, test_x, m, '-b 1')
            return accuary
        #features.map(lambda x:x.split(" ")).map(getmodelandaccuary).repartition(1).saveAsTextFile(self.__savepath)
        features.map(lambda x: x.split(" ")).map(getmodelandaccuary).repartition(1).count()
if __name__ == '__main__':
    stapos.SVMTrainAndPredictOnSpark("hdfs://sunbite-computer:9000/filepath/filepath320240.txt",
                                         "/home/sunbite/accuary").SVMTrainAndPredictOnSpark()