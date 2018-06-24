# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import random


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
    """
    KNN计算准确率
    :param train_X: 训练集特征
    :param train_Y: 训练集标签
    :param test_X: 测试集特征
    :param test_Y: 测试集标签
    :param K: KNN中的K
    :return: predict_Y, accuary, probility： 预测标签，准确率，可能性
    """
    # knn分类器
    knn = KNeighborsClassifier(n_neighbors=K, weights='distance')

    # 训练
    knn.fit(train_X, train_Y)
    # 获得预测可能性
    probility = knn.predict_proba(test_X)
    # 获得准确率
    accuary = knn.score(test_X, test_Y)
    # 获得预测标签
    predict_Y = knn.predict(test_X)

    return predict_Y, accuary, probility


def svm_sklearn(train_X, train_Y, test_X, test_Y):
    """
    SVM计算准确率
    :param train_X: 训练集特征
    :param train_Y: 训练集标签
    :param test_X: 测试集特征
    :param test_Y: 测试集标签
    :return: predict_Y, accuary, probility： 预测标签，准确率，可能性
    """
    # svm分类器
    # 训练
    svc = SVC(C=15, kernel='rbf', degree=3, gamma=2, probability=True)
    svc.fit(train_X, train_Y)
    # 获得预测可能性
    probility = svc.predict_proba(test_X)
    # 获得准确率
    accuary = svc.score(test_X, test_Y)
    # 获得预测标签
    predict_Y = svc.predict(test_X)

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
        Confidence.append(get_confidence_knn(temp))
        idx = np.argmax(kindscounter)
        testResults.put(i, kinds[idx])
    return (Confidence, testResults)


def get_confidence_knn(probility_knn):
    """
    计算KNN置信度
    :param probility_knn: 类别的概率
    :return: confidence_knn：置信度
    """
    if np.size(probility_knn) == 1:
        confidence_knn = 1
    else:
        probility_knn = np.array(probility_knn)
        result = probility_knn[np.argsort(-probility_knn)]
        confidence_knn = result[0] - result[1]

    return confidence_knn


def get_confidence_svm(probility_svm):
    """
    计算SVM置信度
    :param probility_svm: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probility_svm) == 1:
        confidence_svm = 1
    else:
        probility_svm = np.array(probility_svm)
        result = probility_svm[np.argsort(-probility_svm)]
        confidence_svm = result[0] - result[1]
    return confidence_svm


def get_confidence_svm_index(confidence_svm_list, predict_Y_svm, predict_Y_knn, temp_num_svm):
    """
    获取SVM置信度较高的索引
    :param confidence_svm_list:
    :param predict_Y_svm:
    :param predict_Y_knn:
    :param temp_num_svm:
    :return:
    """
    average = int(temp_num_svm / 11)
    #average = int(4)
    index_svm_label_high_confidence = []
    for i in range(1, 11 + 1):
        # 获得knn和svm所预测的在每个类别中的相同标签的索引
        index_same_predictLabel_list = get_same_predictLabel_index_for_each(predict_Y_knn, predict_Y_svm, i)
        # 获得两个分类器所预测的相同标签的对应的置信度
        confidence_svm_same_predictLabel_list = []
        for j in index_same_predictLabel_list:
            confidence_svm_same_predictLabel_list.append(confidence_svm_list[j])
        # 置信度降序排列
        ind_confidence = np.argsort(-np.array(confidence_svm_same_predictLabel_list))
        a = np.array(confidence_svm_same_predictLabel_list)
        b = a[ind_confidence]
        if len(index_same_predictLabel_list) > average:
            top_N = average
        else:
            top_N = len(index_same_predictLabel_list)
        index_same_predictLabel_list = np.array(index_same_predictLabel_list)
        # 按照置信度降序得到相应的排序索引
        index_same_predictLabel_sort_list = index_same_predictLabel_list[ind_confidence]
        # 取前top_N个索引
        index_same_predictLabel_sort_top_list = index_same_predictLabel_sort_list.take(np.arange(0, top_N))
        # 添加到要返回的具有高置信度索引的list
        index_svm_label_high_confidence.extend(index_same_predictLabel_sort_top_list.tolist())
    return index_svm_label_high_confidence


def get_confidence_knn_index(confidence_knn_list, predict_Y_svm, predict_Y_knn, temp_num_knn):
    """
    获取KNN置信度较高的索引
    :param confidence_knn_list:
    :param predict_Y_svm:
    :param predict_Y_knn:
    :param temp_num_knn:
    :return:
    """
    average = int(temp_num_knn / 11)
    #average = int(4)
    index_knn_label_high_confidence = []
    for i in range(1, 11 + 1):
        # 获得knn和svm所预测的在每个类别中的相同标签的索引
        index_same_predictLabel_list = get_same_predictLabel_index_for_each(predict_Y_knn, predict_Y_svm, i)
        # 获得两个分类器所预测的相同标签的对应的置信度
        confidence_knn_same_predictLabel_list = []
        for j in index_same_predictLabel_list:
            confidence_knn_same_predictLabel_list.append(confidence_knn_list[j])
        # 置信度降序排列
        ind_confidence = np.argsort(-np.array(confidence_knn_same_predictLabel_list))
        if len(index_same_predictLabel_list) > average:
            top_N = average
        else:
            top_N = len(index_same_predictLabel_list)
        index_same_predictLabel_list = np.array(index_same_predictLabel_list)
        # 按照置信度降序得到相应的排序索引
        index_same_predictLabel_sort_list = index_same_predictLabel_list[ind_confidence]
        # 取前top_N个索引
        index_same_predictLabel_sort_top_list = index_same_predictLabel_sort_list.take(np.arange(0, top_N))
        # 添加到要返回的具有高置信度索引的list
        index_knn_label_high_confidence.extend(index_same_predictLabel_sort_top_list.tolist())
    return index_knn_label_high_confidence


def get_feature_for_libsvm(featurevalue):
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


def get_feature_from_libsvm(featuredictlist):
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


def get_same_predictLabel_index(predict_Y_knn, predict_Y_svm):
    """
    获得knn和svm的预测标签相同的索引
    :param predict_Y_knn: knn的预测标签
    :param predict_Y_svm: svm的预测标签
    :return:index_same_predictLabel_list knn和svm的预测标签相同的索引list
    """
    index_same_predictLabel_list = []
    if len(predict_Y_knn) == len(predict_Y_svm):
        for i in range(len(predict_Y_knn)):
            if predict_Y_knn[i] == predict_Y_svm[i]:
                index_same_predictLabel_list.append(i)
        return index_same_predictLabel_list
    else:
        return index_same_predictLabel_list


def get_same_predictLabel_index_for_each(predict_Y_knn, predict_Y_svm, k):
    """
    获得knn和svm的预测标签相同的索引
    :param predict_Y_knn: knn的预测标签
    :param predict_Y_svm: svm的预测标签
    :param k: 标签
    :return:index_same_predictLabel_list knn和svm的预测标签相同的索引list
    """
    index_same_predictLabel_list = []
    # if len(predict_Y_knn) == len(predict_Y_svm):
    if len(predict_Y_knn) > len(predict_Y_svm):
        length = len(predict_Y_svm)
    else:
        length = len(predict_Y_knn)
    for i in range(length):
        if predict_Y_knn[i] == predict_Y_svm[i] == k:
            index_same_predictLabel_list.append(i)
    return index_same_predictLabel_list
    # else:
    #     return index_same_predictLabel_list


def get_Y_X_tuple_list(Y_list, X_list):
    """
    返回标签和特征组成的元组list
    :param Y_list: 标签list
    :param X_list: 特征list
    :return:Y_X_tuple_list：标签和特征组成的元组list
    """
    Y_X_tuple_list = []
    if len(Y_list) == len(X_list):
        for i in range(len(Y_list)):
            Y_X_tuple_list.append((Y_list[i], X_list[i]))
        return Y_X_tuple_list
    else:
        Y_X_tuple_list


def get_Y_and_X_list_from_tuple(Y_X_tuple_list):
    """
    从标签和特征组成的元组list得到相应的标签list和特征list
    :param Y_X_tuple_list:标签和特征组成的元组list
    :return:标签list，特征list
    """
    Y_list = []
    X_list = []
    if len(Y_X_tuple_list) is not 0:
        for i in range(len(Y_X_tuple_list)):
            Y_list.append(Y_X_tuple_list[i][0])
            X_list.append(Y_X_tuple_list[i][1])
        return Y_list, X_list
