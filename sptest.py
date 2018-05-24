# -*- coding: utf-8 -*-
from pyspark import SparkContext
import os
import sys
import kafka
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
import VideoSparkStreamingConsumer as vssc
import FeaturesExtractorOnSpark as feos

os.environ['JAVA_HOME'] = r'/home/sunbite/jdk1.8.0_131'
os.environ['SPARK_HOME'] = r'/home/sunbite/spark-2.1.2'
sys.path.append(r"/home/sunbite/spark-2.1.2/python")
sys.path.append("/home/sunbite/jdk1.8.0_131/bin")
# sc = SparkContext(master=r"local", appName=r"VideoSparkStreamingConsumer")
# ssc = StreamingContext(sc, 2)
# messages = KafkaUtils.createDirectStream(ssc=ssc, topics=["test"], kafkaParams={"metadata.broker.list":"10.3.11.131:9092"})
# #decoder = np.vectorize(lambda x: int(x))
# result = messages.map(lambda x:(np.fromstring(x[1],dtype="S3"),1))
# messages.pprint()
# messages.saveAsTextFiles("/home/sunbite/tmp")
# ssc.start()
# ssc.awaitTermination()
sf=vssc.VideoSparkStreamingConsumer(savepath="hdfs://sunbite-computer:9000/keyframe320240/keyframe-soccer_juggling/keyframe-soccer_juggling", topicname=r"test", brokerlist="10.3.11.131:9092")
#sf=vssc.VideoSparkStreamingConsumer(savepath="/home/sunbite/keyframe/keyframe", topicname=r"test", brokerlist="10.3.11.131:9092")
sf.savekeyframe()
#feos.FeaturesExtractorOnSpark("hdfs://sunbite-computer:9000/keyframe/keyframe15256015*/part-00000","hdfs://sunbite-computer:9000/features").extractfeatures()
#feos.FeaturesExtractorOnSpark("/home/sunbite/keyframe/keyframe*/part-00000","hdfs://sunbite-computer:9000/features").extractfeatures()
#feos.FeaturesExtractorOnSpark("/home/sunbite/1").extractfeatures()