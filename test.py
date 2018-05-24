# -*- coding: utf-8 -*-
import KeyFrameExtractor as kfe
import VideoKafkaProducer as vkp

#from pyspark.streaming.kafka import KafkaUtils, TopicAndPartition
import kafka
import GetFeatures1

#k= kfe.KeyFrameExtractor(r"/home/sunbite/video2/v_golf_25_05.avi", r"/home/sunbite/image")
#k.extractkeyframe()
kafkaproducer = vkp.VideoKafkaProducer(r"/home/sunbite/video/action_youtube_naudio/soccer_juggling", r"10.3.11.131:9092", r"test")
#kafkaproducer = vkp.VideoKafkaProducer(r"/home/sunbite/video2", r"10.3.11.131:9092", r"test")
kafkaproducer.start()

# import os
# import shutil
# path = r'/home/sunbite/video/action_youtube_naudio'
# for dirpath,dirnames,filenames in os.walk(path):
#     for filename in filenames:
#         print(os.path.join(dirpath, filename))
#my_feature = GetFeatures.get_features('v_shooting_01_01_0.jpg')
#print(my_feature)