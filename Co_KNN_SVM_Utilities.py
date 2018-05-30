# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


def knn(trainImages, trainLabels, testImages, testLabels, K):
    """
    KNN计算准确率
    :param trainImages: 训练集特征
    :param trainLabels: 训练集标签
    :param testImages: 测试集特征
    :param testLabels: 测试集标签
    :param K: KNN中的K
    :return: num： 错误率
    """
    testLength = np.array(testImages).shape[0]
    # 保存测试集的预测结果
    testResults = np.linspace(0, 0, testLength)
    # 保存预测置信度
    Confidence = []

    Confidence, testResults = getConfidenceAndTestResults(trainImages, trainLabels, testImages, K)

    # 计算错误率
    error = 0
    for i in range(0, testLength):
        if (testResults[i] != testLabels[i]):
            error = error + 1
    accuary = 1 - error / testLength

    return testResults, accuary, Confidence

def knn_sklearn(train_X, train_Y, test_X, test_Y, K):
    knn = KNeighborsClassifier(n_neighbors = K,weights='distance')
    knn.fit(train_X, train_Y)
    probility = knn.predict_proba(test_X)
    accuary = knn.score(test_X, test_Y)
    predict_Y = knn.predict(test_X)

    return predict_Y, accuary, probility

def getConfidenceAndTestResults(trainImages, trainLabels, testImages, K):
    """
    获取KNN置信度和测试
    :param trainImages: 训练集特征
    :param trainLabels: 训练集标签
    :param testImages: 测试集特征
    :param testLabels: 测试集标签
    :param K: KNN中的K
    :return:
    """
    trainImages = np.array(trainImages)
    testImages = np.array(testImages)
    # 计算训练集的长度和测试集的长度
    trainLength = trainImages.shape[0]
    testLength = testImages.shape[0]
    # 保存测试集的预测结果
    testResults = np.linspace(0, 0, testLength)
    compLabel = np.linspace(0, 0, K)
    # 保存预测置信度
    Confidence = []

    for i in range(0, testLength):
        curImage = np.tile(testImages[i, :], (trainLength, 1))
        curImage = np.abs(trainImages - curImage)
        comp = np.sum(curImage, 1)
        sortedComp = np.sort(comp)
        ind = np.argsort(comp)
        for j in range(0, K):
            compLabel.put(j, trainLabels[ind[j]])
        table = Counter(compLabel)
        # 计算每个测试样本的预测置信度
        # 类别的计数
        kindscounter = np.array(list(table.values()))
        # 类别
        kinds = np.array(list(table.keys()))
        # 类别的概率
        temp = kindscounter / compLabel.size
        Confidence.append(sub(temp))
        idx = np.argmax(kindscounter)
        testResults.put(i, kinds[idx])
    return (Confidence, testResults)


def sub(temp):
    """
    计算KNN置信度
    :param temp: 类别的概率
    :return: sub_num：置信度
    """
    if np.size(temp) == 1:
        sub_num = 1
    else:
        temp = np.array(temp)
        result = temp[np.argsort(-temp)]
        sub_num = result[0] - result[1]
        #sub_num = result[0]

    return sub_num


def sub_SVM(temp):
    """
    计算SVM置信度
    :param temp: 类别的概率
    :return: sub_num_SVM 置信度
    """
    if np.size(temp) == 1:
        sub_num_SVM = 1
    else:
        temp = np.array(temp)
        result = temp[np.argsort(-temp)]
        sub_num_SVM = result[0] - result[1]
        #sub_num_SVM = result[0]
    return sub_num_SVM


def getConfidence_SVM(Confidence_SVM, predict_label, temp_num, testLength):
    """
    获取SVM置信度
    :param Confidence_SVM:
    :param predict_label:
    :param temp_num:
    :param testLength:
    :return:
    """
    # 保存置信度大于0.9的索引值
    temp_label_SVM = []
    # 计数
    temp_label_index_SVM = 0

    ind_confidence = np.argsort(np.array(Confidence_SVM))
    # for i in range(0, testLength):
    #     if Confidence_SVM[i] > 0:
    #         temp_label_index_SVM = temp_label_index_SVM + 1
    #         temp_label_SVM[temp_label_index_SVM] = i
    temp_label_SVM = ind_confidence
    # svm_label代表的是索引号temp_label所处位置的标签
    svm_label = []
    for i in temp_label_SVM:
        svm_label.append(predict_label[i])

    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    num6 = 0
    num7 = 0
    num8 = 0
    num9 = 0
    num10 = 0
    num11 = 0
    average = temp_num / 11
    temp_label_index = 0
    total = np.array(temp_label_SVM).shape[0]

    index_label_SVM = []

    # for i in range(0,temp_num):
    #     index_label_SVM.append(temp_label_SVM[i])

    for j in range(0, total):
        if (svm_label[j] == 1) and (num1 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num1 = num1 + 1
        elif (svm_label[j] == 2) and (num2 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num2 = num2 + 1
        elif (svm_label[j] == 3) and (num3 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num3 = num3 + 1
        elif (svm_label[j] == 4) and (num4 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num4 = num4 + 1
        elif (svm_label[j] == 5) and (num5 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num5 = num5 + 1
        elif (svm_label[j] == 6) and (num6 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num6 = num6 + 1
        elif (svm_label[j] == 7) and (num7 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num7 = num7 + 1
        elif (svm_label[j] == 8) and (num8 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num8 = num8 + 1
        elif (svm_label[j] == 9) and (num9 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num9 = num9 + 1
        elif (svm_label[j] == 10) and (num10 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num10 = num10 + 1
        elif (svm_label[j] == 11) and (num11 < average):
            index_label_SVM.append(temp_label_SVM[j])
            num11 = num11 + 1

        if (np.array(index_label_SVM).shape[0] == temp_num):
            break
    return index_label_SVM


def getConfidence(Confidence, testResults, temp_num):
    """
    从被预测的伪标签中选择200个置信度高的测试样本，每类20个（temp_num/10）
    :param Confidence:
    :param testResults:
    :param temp_num:
    :return:
    """
    total = np.array(Confidence).shape[0]
    # 保存200个索引号
    temp_label = []
    # 根据置信度排序 ind_confidence表示索引号
    ind_confidence = np.argsort(np.array(Confidence))
    # 每一个索引对应的预测标签
    temp_total = []
    for i in ind_confidence:
        temp_total.append(testResults[i])
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    num6 = 0
    num7 = 0
    num8 = 0
    num9 = 0
    num10 = 0
    num11 = 0
    average = temp_num / 11
    # for i in range(0,temp_num):
    #     temp_label.append(ind_confidence[i])
    for j in range(0, total):
        if (temp_total[j] == 1) and (num1 < average):
            temp_label.append(ind_confidence[j])
            num1 = num1 + 1
        elif (temp_total[j] == 2) and (num2 < average):
            temp_label.append(ind_confidence[j])
            num2 = num2 + 1
        elif (temp_total[j] == 3) and (num3 < average):
            temp_label.append(ind_confidence[j])
            num3 = num3 + 1
        elif (temp_total[j] == 4) and (num4 < average):
            temp_label.append(ind_confidence[j])
            num4 = num4 + 1
        elif (temp_total[j] == 5) and (num5 < average):
            temp_label.append(ind_confidence[j])
            num5 = num5 + 1
        elif (temp_total[j] == 6) and (num6 < average):
            temp_label.append(ind_confidence[j])
            num6 = num6 + 1
        elif (temp_total[j] == 7) and (num7 < average):
            temp_label.append(ind_confidence[j])
            num7 = num7 + 1
        elif (temp_total[j] == 8) and (num8 < average):
            temp_label.append(ind_confidence[j])
            num8 = num8 + 1
        elif (temp_total[j] == 9) and (num9 < average):
            temp_label.append(ind_confidence[j])
            num9 = num9 + 1
        elif (temp_total[j] == 10) and (num10 < average):
            temp_label.append(ind_confidence[j])
            num10 = num10 + 1
        elif (temp_total[j] == 11) and (num11 < average):
            temp_label.append(ind_confidence[j])
            num11 = num11 + 1

        if (np.array(temp_label).shape[0] == temp_num):
            break
    return temp_label


def getfeatureforlibsvm(featurevalue):
    """
    把常规的特征list转换成libsvm需要的特征
    :param featurevalue: 常规的特征list
    :return: 返回[{1：aaa,2:bbb,...},...]形式的特征list
    """
    featuredictlist = []
    for i in range(0, len(featurevalue)):
        featuremap = {}
        for j in range(0, len(featurevalue[i])):
            featuremap[j + 1] = featurevalue[i][j]
        featuredictlist.append(featuremap)
    return featuredictlist


def getfeaturefromlibsvm(featuredictlist):
    """
    返回特征list
    :param featuredictlist: libsvm的feature形式
    :return:返回[aaa,bbb,...]形式的特征list
    """
    featurelists = []
    for i in range(0, len(featuredictlist)):
        featurelist = []
        for j in range(1, len(featuredictlist[i]) + 1):
            featurelist.append(featuredictlist[i][j])
        featurelists.append(featurelist)
    return featurelists
