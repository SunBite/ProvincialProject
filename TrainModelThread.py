# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import time
import re
import paramiko


class TrainModelThread(QThread):
    # 线程信号
    sendlog = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self.stopped = True
        self.mutex = QMutex()
        self.func = func

    def run(self):
        """
        线程开始运行
        :return:
        """
        # 互斥锁
        with QMutexLocker(self.mutex):
            self.stopped = False
        if not self.stopped:
            self.sendlog.connect(self.func)
            self.trainModel(self.sendlog)
        else:
            return

    def is_stopped(self):
        """
        线程状态是否是停止
        :return:
        """
        with QMutexLocker(self.mutex):
            return self.stopped

    def stop(self):
        """
        线程停止
        :return:
        """
        # 互斥锁
        with QMutexLocker(self.mutex):
            self.stopped = True

    def connetServer(self):
        """
        连接server
        :return:
        """
        hostname = '10.3.11.131'
        username = 'sunbite'
        password = 'yinyu1988318'
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname=hostname, username=username, password=password)
            stdin, stdout, stderr = self.ssh.exec_command('who')
            search = re.search(stdout.read().decode(), username)
            if search:
                info = '服务器连接成功。'
            else:
                info = '服务器连接失败。'

        except:
            info = '服务器连接失败。'
        return info

    def deleteAndCreateTmp(self):
        """
        删除临时文件
        :return:
        """
        # 删除关键帧文件夹
        rm_keyframe_stdin, rm_keyframe_stdout, rm_keyframe_stderr = self.ssh.exec_command('rm -r /home/sunbite/Co_KNN_SVM_TMP/keyframe')
        print(rm_keyframe_stdout.read())
        print(rm_keyframe_stderr.read())
        if rm_keyframe_stderr:
            # 创建空的关键帧文件夹
            mk_keyframe_stdin, mk_keyframe_stdout, mk_keyframe_stderr = self.ssh.exec_command('mkdir /home/sunbite/Co_KNN_SVM_TMP/keyframe')
            print(mk_keyframe_stdout.read())
            print(mk_keyframe_stderr.read())
        # 删除关键帧提取准备的参数文件
        rm_videopath_stdin, rm_videopath_stdout, rm_videopath_stderr = self.ssh.exec_command(
            'rm /home/sunbite/Co_KNN_SVM_TMP/videopath.txt')
        print(rm_videopath_stdout.read())
        print(rm_videopath_stderr.read())
        # 删除特征提取准备的参数文件
        rm_keyframepath_stdin, rm_keyframepath_stdout, rm_keyframepath_stderr = self.ssh.exec_command(
            'rm /home/sunbite/Co_KNN_SVM_TMP/keyframepath.txt')
        print(rm_keyframepath_stdout.read())
        print(rm_keyframepath_stderr.read())
        # 删除特征提取文件夹
        rm_features_stdin, rm_features_stdout, rm_features_stderr = self.ssh.exec_command(
            'rm -r /home/sunbite/Co_KNN_SVM_TMP/features320240-366')
        print(rm_features_stdout.read())
        print(rm_features_stderr.read())
        # if rm_features_stderr:
        #     # 创建特征提取文件夹
        #     mk_features_stdin, mk_features_stdout, mk_features_stderr = self.ssh.exec_command(
        #         'mkdir /home/sunbite/Co_KNN_SVM_TMP/features320240-366')
        #     print(mk_features_stdout.read())
        #     print(mk_features_stderr.read())
        # 删除协同训练保存的model文件
        rm_keyframepath_stdin, rm_keyframepath_stdout, rm_keyframepath_stderr = self.ssh.exec_command(
            'rm /home/sunbite/Co_KNN_SVM_TMP/CoKNNSVM.model')
        print(rm_keyframepath_stdout.read())
        print(rm_keyframepath_stderr.read())

    def trainModel(self, sendlog):
        """
        模型训练
        :param sendlog:要发送的log信号
        :return:
        """
        self.connetServer()
        #self.deleteAndCreateTmp()
        # 关键帧提取参数准备
        # videopathwriter_stdin, videopathwriter_stdout, videopathwriter_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;python /home/sunbite/PycharmProjects/myApp/VideoPathWriter.py')
        # print(videopathwriter_stdout.read())
        # # 关键帧提取
        # keyframeextractoronspark_stdin, keyframeextractoronspark_stdout, keyframeextractoronspark_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myApp/KeyFrameExtractor.py --master local[2] /home/sunbite/PycharmProjects/myApp/KeyFrameExtractorOnSpark.py')
        # print(keyframeextractoronspark_stdout.read())
        # # 特征提取参数准备
        # keyframepathwriter_stdin, keyframepathwriter_stdout, keyframepathwriter_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;python /home/sunbite/PycharmProjects/myApp/KeyFramePathWriter.py')
        # print(keyframepathwriter_stdout.read())
        # # 特征提取
        # featuresextractoronspark_stdin, featuresextractoronspark_stdout, featuresextractoronspark_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myApp/GetFeatures.py --master local[2] /home/sunbite/PycharmProjects/myApp/FeaturesExtractorOnSpark.py')
        # print(featuresextractoronspark_stdout.read())
        # # 协同训练保存model
        # coknnsvmtrainandpredictonspark_stdin, coknnsvmtrainandpredictonspark_stdout, coknnsvmtrainandpredictonspark_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myApp/ListParam.py,/home/sunbite/PycharmProjects/myApp/Co_KNN_SVM_New.py,/home/sunbite/PycharmProjects/myApp/Co_KNN_SVM_Utilities.py --master local[2] /home/sunbite/PycharmProjects/myApp/CoKNNSVMTrainAndPredictOnSpark.py')
        # print(coknnsvmtrainandpredictonspark_stdout.read())
        time.sleep(20)
        sendlog.emit("模型训练完成。")
