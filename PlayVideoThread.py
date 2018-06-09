# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import time


class PlayVideoThread(QThread):
    # 线程信号
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stopped = True
        self.mutex = QMutex()
        self.fps = 30

    def run(self):
        """
        线程开始运行
        :return:
        """
        # 互斥锁
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            # 发射信号
            self.signal.emit("1")
            time.sleep(1 / self.fps)

    def setFps(self, fps):
        """
        设置fps
        :param fps:
        :return:
        """
        self.fps = fps

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
