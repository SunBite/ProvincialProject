# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, qApp, QFileDialog, QHeaderView
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import QRect, QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
import os
import cv2
import time
import PlayVideoThread as pvt


class Co_KNN_SVM_UI(QMainWindow):

    def __init__(self):
        super().__init__()
        # 初始化UI
        self.initUI()
        # 创建action
        self.initAction()
        self.center()
        self.setWindowTitle("科海人体行为动作识别系统")
        self.setFixedSize(750, 570)

    def initUI(self):
        """
        初始化UI
        :return:
        """
        # 状态栏
        self.statusBar()

        # 菜单栏
        self.menubar = self.menuBar()

        # 打开menu
        self.menu_openvideos = QtWidgets.QMenu()
        self.menu_openvideos.setTitle("打开")
        self.menubar.addMenu(self.menu_openvideos)

        # 处理menu
        self.menu_dispose = QtWidgets.QMenu()
        self.menu_dispose.setTitle("处理")
        self.menubar.addMenu(self.menu_dispose)

        # 帮助menu
        self.menu_help = QtWidgets.QMenu()
        self.menu_help.setTitle("帮助")
        self.menubar.addMenu(self.menu_help)

        # 检测视频文件目录label
        self.label_videofiledir = QtWidgets.QLabel(self)
        self.label_videofiledir.setText("视频目录:")
        self.label_videofiledir.move(30, 30)

        # 检测视频文件目录
        self.treeview_videofiledir = QtWidgets.QTreeView(self)
        self.treeview_videofiledir.resize(340, 480)
        self.treeview_videofiledir.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.treeview_videofiledir.setSizeAdjustPolicy(QtWidgets.QTreeView.AdjustToContents)
        self.treeview_videofiledir.header().setFixedWidth(700)
        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderItem(0, QtGui.QStandardItem("视频列表"))
        self.treeview_videofiledir.setModel(self.model)
        self.treeview_videofiledir.move(30, 60)

        # 播放视频的线程
        #self.playvideothread = pvt.PlayVideoThread()

        # 双击文件目录连接启动播放视频的线程
        self.treeview_videofiledir.doubleClicked.connect(self.startPlayVideoThread)

        # 视频播放label
        self.label_videoplay = QtWidgets.QLabel(self)
        self.label_videoplay.setText("视频播放：")
        self.label_videoplay.move(400, 30)

        # 显示视频框
        self.picturelabel = QtWidgets.QLabel(self)
        init_image = QPixmap("no_video.jpg").scaled(320, 240)
        self.picturelabel.setPixmap(init_image)
        self.picturelabel.resize(320, 240)
        self.picturelabel.move(400, 60)

        # 显示处理信息label
        self.label_showmessage = QtWidgets.QLabel(self)
        self.label_showmessage.setText("显示处理信息：")
        self.label_showmessage.move(400, 310)
        # 显示处理信息框
        self.textedit = QtWidgets.QTextEdit(self)
        self.textedit.setReadOnly(True)
        self.textedit.resize(320, 200)
        self.textedit.move(400, 340)

    def initAction(self):
        """
        创建action
        :return:
        """
        # 选取待检测视频menu的Action
        self.action_openvideos_choosevideos = QtWidgets.QAction("选取视频", self)
        self.action_openvideos_choosevideos.triggered.connect(self.chooseVideos)
        self.action_openvideos_choosefolder = QtWidgets.QAction("选取文件夹", self)
        self.action_openvideos_choosefolder.triggered.connect(self.chooseFolder)
        self.action_openvideos_clear = QtWidgets.QAction("清空", self)
        self.action_openvideos_clear.triggered.connect(self.openVideos_clear)
        self.menu_openvideos.addAction(self.action_openvideos_choosevideos)
        self.menu_openvideos.addAction(self.action_openvideos_choosefolder)
        self.menu_openvideos.addAction(self.action_openvideos_clear)
        self.menu_openvideos.addSeparator()
        # 处理menu的Action
        self.action_modeltrain = QtWidgets.QAction("模型训练", self, triggered=qApp.quit)
        self.action_detect = QtWidgets.QAction("检测", self, triggered=qApp.quit)
        self.menu_dispose.addAction(self.action_modeltrain)
        self.menu_dispose.addAction(self.action_detect)
        self.menu_dispose.addSeparator()
        # 帮助menu的Action
        self.action_lookuphelp = QtWidgets.QAction("帮助", self, triggered=qApp.quit)
        self.action_about = QtWidgets.QAction("关于", self, triggered=qApp.quit)
        self.menu_help.addAction(self.action_lookuphelp)
        self.menu_help.addAction(self.action_about)
        self.menu_help.addSeparator()

    def center(self):
        """
        控制窗口显示在屏幕中心的方法
        :return:
        """
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        # self.move(qr.topLeft())
        self.move(320, 70)

    def chooseVideos(self):
        """
        选取要检测的视频加入到视频列表中
        :return:
        """
        files, filetype = QFileDialog.getOpenFileNames(None, "视频选择", os.path.expanduser("~"),
                                                       "Video Files (*.avi *.mpg)")
        if files:
            # 清空model
            self.openVideos_clear()
            videoItem = QtGui.QStandardItem(os.path.dirname(files[0]))
            videoItem.setEditable(False)
            for file in files:
                videoItemChild = QtGui.QStandardItem(os.path.basename(file))
                videoItemChild.setEditable(False)
                videoItem.appendRow(videoItemChild)
            self.model.appendRow(videoItem)
            self.treeview_videofiledir.setModel(self.model)

    def chooseFolder(self):
        """
        把文件夹中的视频加入视频列表中
        :return:
        """
        directory = QFileDialog.getExistingDirectory(None, "文件夹选择", os.path.expanduser("~"))
        if os.path.isdir(directory):
            # 清空model
            self.openVideos_clear()
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(directory):
                if filenames:
                    videoItem = QtGui.QStandardItem(dirpath)
                    videoItem.setEditable(False)
                    for filename in filenames:
                        # 获取文件名后缀，通过os.path.splitext（path）分离文件名和扩展名
                        ext = os.path.splitext(filename)[1]
                        # 将文件名统一转化为小写
                        ext = ext.lower()
                        if ext == '.avi' or ext == '.mpg':
                            videoItemChild = QtGui.QStandardItem(os.path.basename(filename))
                            videoItemChild.setEditable(False)
                            videoItem.appendRow(videoItemChild)
                    self.model.appendRow(videoItem)
                    self.treeview_videofiledir.setModel(self.model)

    def openVideos_clear(self):
        """
        清空视频列表
        :return:
        """
        self.model.clear()
        headerItem = QtGui.QStandardItem("视频列表")
        headerItem.setEditable(False)
        self.model.setHorizontalHeaderItem(0, headerItem)
        self.treeview_videofiledir.setModel(self.model)
        init_image = QPixmap("no_video.jpg").scaled(320, 240)
        self.picturelabel.setPixmap(init_image)

    def startPlayVideoThread(self):
        """
        启动播放视频的线程
        :return:
        """
        # 获取选中的当前的Index
        index = self.treeview_videofiledir.selectionModel().currentIndex()
        # 如果当前Index有父节点，也就是说点击视频文件名
        if index.parent().data():
            # 视频的路径
            videopath = index.parent().data() + os.sep + index.data()
            self.playCapture = cv2.VideoCapture(videopath)
            fps = self.playCapture.get(cv2.CAP_PROP_FPS)
            self.timer = QTimer()
            self.timer.timeout.connect(self.playVideo)
            self.timer.start(1000 / fps)
            # self.playvideothread.setFps(fps)
            # # 连接播放视频槽
            # self.playvideothread.signal.connect(self.playVideo)
            # self.playvideothread.start()
    def playVideo(self):
        """
        播放视频
        :return:
        """
        if self.playCapture.isOpened():
            ret, frame = self.playCapture.read()
            if ret:
                #self.treeview_videofiledir.setDisabled(True)
                # 获取视频播放label的大小
                s = self.picturelabel.rect()
                # frame重置大小
                R_frame = cv2.resize(frame, (QRect.width(s), QRect.height(s)))
                if R_frame.ndim == 3:
                    R_frame_RGB = cv2.cvtColor(R_frame, cv2.COLOR_BGR2RGB)
                elif R_frame.ndim == 2:
                    R_frame_RGB = cv2.cvtColor(R_frame, cv2.COLOR_GRAY2BGR)
                qImage = QtGui.QImage(R_frame_RGB[:], R_frame_RGB.shape[1], R_frame_RGB.shape[0],
                                      QtGui.QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImage)
                self.picturelabel.setPixmap(pixmap)
            else:
                # 释放VideoCapture
                self.playCapture.release()
                # 关闭线程
                #self.playvideothread.stop()
                #self.treeview_videofiledir.setDisabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Co_KNN_SVM_UI()
    w.show()
    sys.exit(app.exec_())
