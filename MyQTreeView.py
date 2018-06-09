# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QTreeView, QFileSystemModel, QApplication, QDirModel
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
from PyQt5.QtCore import QThread, QRect
import PlayVideoThread as pvt


class MyQTreeView(QTreeView):

    def __init__(self, parent=None):
        super(MyQTreeView, self).__init__(parent)
        self.__videolist = []

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):
        """
        重写mouseDoubleClickEvent方法，得到双击的视频路径
        :param e:
        :return:
        """
        index = self.selectionModel().currentIndex()
        if index.parent().data():
            videopath = index.parent().data() + os.sep + index.data()
            self.__videolist.clear()
            self.__videolist.append(videopath)

    def getvideolist(self):
        return self.__videolist
