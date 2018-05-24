# -*- coding: utf-8 -*-
from pyspark import SparkContext
import pyspark.mllib.classification
import os
import sys
import random
import kafka
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
import VideoSparkStreamingConsumer as vssc
import FeaturesExtractorOnSpark as feos
from pyspark.accumulators import  AccumulatorParam
from ListParam import *
import svmutil
import SaveAsLibSVMFile as salf
import SVMTrainAndPredictOnSpark as stapos

os.environ['JAVA_HOME'] = r'/home/sunbite/jdk1.8.0_131'
os.environ['SPARK_HOME'] = r'/home/sunbite/spark-2.1.2'
sys.path.append(r"/home/sunbite/spark-2.1.2/python")
sys.path.append("/home/sunbite/jdk1.8.0_131/bin")

# # class ListParam(AccumulatorParam):
# #     def zero(self, value):
# #         return []
# #     def addInPlace(self, value1, value2):
# #         value1.extend(value2)
# #         return value1
#
# sc = SparkContext(master="local[2]",appName="sptest3")
# rdd1 = sc.textFile(r"hdfs://sunbite-computer:9000/features/part-*")
# #rdd1 = sc.textFile(r"/home/sunbite/part-00000-1")
#
#
# classmap = {"basketball": 1, "biking": 2, "diving": 3, "golf_swing": 4, "horse_riding": 5, "soccer_juggling": 6,
#
#                            "swing": 7, "tennis_swing": 8, "trampoline_jumping": 9, "volleyball_spiking": 10, "walking": 11}
#
# TOTALFEATUREANDLABEL= sc.accumulator([], ListParamForFeatureAndLabel())
#
#
#
# def codechange(line):
#     classname = os.path.basename(line[0]).split("_v")[0]
#     classnum = classmap[classname]
#     ResultIterable = list(line[1])
#     #features = ResultIterable[0] + ResultIterable[1] + ResultIterable[2]
#     #ndarrayfeatures = np.array(ResultIterable)
#     #strtofloat64decoder = np.vectorize(lambda x: np.float(x))
#     #ndarrayfeatures = strtofloat64decoder(ndarrayfeatures)
#     #print(classnum)
#     #print(ResultIterable)
#     #print(ndarrayfeatures.dtype)
#     #print(ndarrayfeatures.size)
#     #print(ndarrayfeatures)
#     return (classnum, ResultIterable)
#
# def getfeaturesandlabel(line):
#     global TOTALFEATUREANDLABEL
#
#     TOTALFEATUREANDLABEL += [(line[0],line[1])]
#     return TOTALFEATUREANDLABEL
# def gettrainandpredict(line):
#     print(line[0])
#     return line
# #rdd2= rdd1.map(lambda x:x.split(" ")).map(lambda x :x[0])
# rdd3= rdd1.map(lambda x:x.split(" ")).map(lambda x:(x[1],x[2:])).map(codechange).map(getfeaturesandlabel)
# #rdd3= rdd1.map(lambda x :x[4])
#
#
#
# #print(rdd2.collect()
# rdd3.count()
# totalfeatureandlabel = TOTALFEATUREANDLABEL.value
#
# def getfeaturelistandlabellist(totalfeatureandlabel):
#     TOTALFEATURE = []
#     TOTALFEATURELABEL = []
#     for i in range(0,len(totalfeatureandlabel)):
#         TOTALFEATURELABEL.append(totalfeatureandlabel[i][0])
#         TOTALFEATURE.append(totalfeatureandlabel[i][1])
#     return (TOTALFEATURELABEL,TOTALFEATURE)
#
# totalfeaturelabel ,totalfeature = getfeaturelistandlabellist(totalfeatureandlabel)
#
# randomindex = [i for i in range(len(totalfeatureandlabel))]
# random.shuffle(randomindex)
# totalfeature = [totalfeature[x] for x in randomindex]
# totalfeaturelabel = [totalfeaturelabel[x] for x in randomindex]
#
# trainfeaturevalue = totalfeature[0:880]
# #print(trainfeaturevalue)
# predictfeaturevalue = totalfeature[880:1100]
#
# def getfeatureforlibsvm(featurevalue):
#     featuredictlist=[]
#     #featurevalue = [list(x) for x in featurevalue]
#     for i in range(0,len(featurevalue)):
#         featuredickkey = list(range(1,len(featurevalue[i])+1))
#         featuredickvalue = [float(x) for x in featurevalue[i]]
#         featuredict = dict(zip(featuredickkey, featuredickvalue))
#         featuredictlist.append(featuredict)
#     return featuredictlist
#
# trainfeaturevalueforlibsvm = getfeatureforlibsvm(trainfeaturevalue)
# #print(trainfeaturevalueforlibsvm)
# predictfeaturevalueforlibsvm = getfeatureforlibsvm(predictfeaturevalue)
#
# trainfeaturelabel = totalfeaturelabel[0:880]
# predictfeaturelabel = totalfeaturelabel[880:1100]
# #print(predictfeaturelabel)
# trainfeaturelabel = [float(x) for x in trainfeaturelabel]
# predictfeaturelabel = [float(x) for x in predictfeaturelabel]
# #print(predictfeaturelabel)
# model = svmutil.svm_train(trainfeaturelabel,trainfeaturevalueforlibsvm,'-s 0 -t 2 -c 5.6569 -g 2 -b 0')
# #model = svmutil.svm_train(trainfeaturelabel,trainfeaturevalueforlibsvm)
# predict_label, accuary, prob_estimates = svmutil.svm_predict(predictfeaturelabel, predictfeaturevalueforlibsvm, model, '-b 0')
# #predict_label, accuary, prob_estimates = svmutil.svm_predict(predictfeaturelabel, predictfeaturevalueforlibsvm, model)
# print(accuary)
# classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
# for i in classname:
#      print(i)
#      salf.SaveAsLibSVMFile("hdfs://sunbite-computer:9000/features64/features-"+i,"/home/sunbite/libsvmfile/"+i).saveaslibSVMfile()
# classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
# featuresandlabel={}
# train_y=[]
# train_x=[]
# test_y=[]
# test_x=[]
# for i in classname:
#      libsvmfilepath = "/home/sunbite/libsvmfile/"+ i +  "/part-00000"
#      featuresandlabel[i] = svmutil.svm_read_problem(libsvmfilepath)
#      print(len(featuresandlabel[i][0]))
#      train_y.extend(featuresandlabel[i][0][0:60])
#      train_x.extend(featuresandlabel[i][1][0:60])
#      test_y.extend(featuresandlabel[i][0][60:300])
#      test_x.extend(featuresandlabel[i][1][60:300])
# print(len(train_y))
# print(len(train_x))
# print(len(test_y))
# print(len(test_x))
# print(train_x)
# m = svmutil.svm_train(train_y, train_x,"-s 0 -t 2 -c 32 -g 8 -b 1")
# predict_label, accuary, prob_estimates = svmutil.svm_predict(test_y, test_x, m, '-b 1')
#train_y, train_x = svmutil.svm_read_problem(r"/home/sunbite/1")
#print(train_y)
#print(train_x)
#test_y, test_x = svmutil.svm_read_problem(ibsvmfiletestpath)
# sc = SparkContext(master="local[2]",appName="sptest3")
# rdd1 = sc.textFile(r"/home/sunbite/1")
# TOTALFEATURE= sc.accumulator([], ListParamForFeatureAndLabel())
# TOTALLABEL= sc.accumulator([], ListParamForFeatureAndLabel())
# def a(line):
#     global TOTALFEATURE
#     global TOTALLABEL
#     prob_y = []
#     prob_x = []
#     line = line.split(None, 1)
#     if len(line) == 1: line += ['']
#     label, features = line
#     xi = {}
#     for e in features.split():
#         ind, val = e.split(":")
#         xi[int(ind)] = float(val)
#     prob_y += [float(label)]
#     prob_x += [xi]
#     TOTALLABEL += prob_y
#     TOTALFEATURE += prob_x
#     return TOTALLABEL, TOTALFEATURE
# def b(x):
#     print(x)
#     model = svmutil.svm_train(x[0].value, x[1].value, '-s 0 -t 2 -c 5.6569 -g 2 -b 0')
# rdd1.map(a).map(b).collect()
#stapos.SVMTrainAndPredictOnSpark("hdfs://sunbite-computer:9000/filepath/filepath1.txt","/home/sunbite/accuary").SVMTrainAndPredictOnSpark()