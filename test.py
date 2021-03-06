# -*- coding: utf-8 -*-
import KeyFrameExtractor as kfe
import VideoKafkaProducer as vkp
import re
import paramiko
import VideoDetector
from sklearn import datasets
from sklearn.model_selection import train_test_split

# from pyspark.streaming.kafka import KafkaUtils, TopicAndPartition
import kafka
import GetFeatures1

# k= kfe.KeyFrameExtractor(r"/home/sunbite/video2/v_golf_25_05.avi", r"/home/sunbite/image")
# k.extractkeyframe()
# kafkaproducer = vkp.VideoKafkaProducer(r"/home/sunbite/video/action_youtube_naudio/soccer_juggling", r"10.3.11.131:9092", r"test")
# kafkaproducer = vkp.VideoKafkaProducer(r"/home/sunbite/video2", r"10.3.11.131:9092", r"test")
# kafkaproducer.start()

# import os
# import shutil
# path = r'/home/sunbite/video/action_youtube_naudio'
# for dirpath,dirnames,filenames in os.walk(path):
#     for filename in filenames:
#         print(os.path.join(dirpath, filename))
# my_feature = GetFeatures.get_features('v_shooting_01_01_0.jpg')
# print(my_feature)
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(hostname='10.3.11.131', username='sunbite', password='yinyu1988318')
# stdin, stdout, stderr = ssh.exec_command('who')
# print()
# print(stdout.read())
# vd = VideoDetector.VideoDetector("/home/sunbite/video/action_youtube_naudio/basketball/v_shooting_01_01.avi","/home/sunbite/Co_KNN_SVM_TMP/CoKNNSVM.model")
# print(vd.getLabel())
x,y=datasets.load_svmlight_file("/home/sunbite/dataset/dataset")
print(x.todense())
print(y)

train_x,test_x,train_y,test_y=train_test_split(x.todense(),y,test_size=0.2,random_state=42,
                                                                                          shuffle=False)
print(len(train_x))
print(len(test_x))
print(len(train_y))
print(len(test_y))