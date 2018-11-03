# -*- coding: utf-8 -*-
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import random


class DataPreparation:
    def __init__(self):
        self.__classmap = {"basketball": 1, "biking": 2, "diving": 3, "golf_swing": 4, "horse_riding": 5,
                           "soccer_juggling": 6, "swing": 7, "tennis_swing": 8, "trampoline_jumping": 9,
                           "volleyball_spiking": 10, "walking": 11}
        # self.__Label_Name_train_tuple_list, self.__Label_Name_test_tuple_list = self.__getLabelAndNameTupleList(path)

    def getLabelAndNameTupleList(self, path):
        """
        遍历文件夹下所有的feature，获取到split之后的名字列,得到相应的标签
        :param path:文件夹路径
        :return: 有标签数据的标签和名字Tuplelist，无标签数据的标签和名字Tuplelist，测试集数据的标签和名字Tuplelist
        """

        classnames = ["basketball", "biking", "diving", "golf_swing", "horse_riding",
                      "soccer_juggling", "swing", "tennis_swing", "trampoline_jumping",
                      "volleyball_spiking", "walking"]
        testandtrains = ["test", "train"]
        trainnameList = []
        trainlabelList = []
        testnameList = []
        testlabelList = []
        for classname in classnames:
            for testandtrain in testandtrains:
                if testandtrain is "train":
                    featurePath = path + classname + os.sep + testandtrain + os.sep + "hogvideoFeature"
                    # 遍历文件夹
                    for dirpath, dirnames, filenames in os.walk(featurePath):
                        for filename in filenames:
                            labelIndex = self.__classmap[filename.split("_v")[0]]
                            trainnameList.append(filename)
                            trainlabelList.append(labelIndex)
                if testandtrain is "test":
                    featurePath = path + classname + os.sep + testandtrain + os.sep + "hogvideoFeature"
                    # 遍历文件夹
                    for dirpath, dirnames, filenames in os.walk(featurePath):
                        for filename in filenames:
                            labelIndex = self.__classmap[filename.split("_v")[0]]
                            testnameList.append(filename)
                            testlabelList.append(labelIndex)
        # 训练集和测试集比例为9：1
        unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(trainnameList, trainlabelList,
                                                                          test_size=0.2,
                                                                          stratify=trainlabelList, random_state=2)  # 2 0.234
        # 有标签数据的标签和名字的Tuplelist
        Label_Name_Labeled_train_tuple_list = self.__get_Y_X_tuple_list(labeled_y, labeled_x)
        # 无标签数据的标签和名字Tuplelist
        Label_Name_Unlabeled_train_tuple_list = self.__get_Y_X_tuple_list(unlabeled_y, unlabeled_x)

        random_index = [i for i in range(len(testnameList))]
        random.shuffle(random_index)
        test_x = [testnameList[x] for x in random_index]
        test_y = [testlabelList[x] for x in random_index]

        # 测试集数据的标签和名字Tuplelist
        Label_Name_test_tuple_list = self.__get_Y_X_tuple_list(test_y, test_x)
        return Label_Name_Labeled_train_tuple_list, Label_Name_Unlabeled_train_tuple_list, Label_Name_test_tuple_list

        # 1 == 1
        #
        # if os.path.isdir(path):
        #     # 遍历文件夹
        #     for dirpath, dirnames, filenames in os.walk(path):
        #         for filename in filenames:
        #             labelIndex = self.__classmap[filename.split("_v")[0]]
        #             nameList.append(filename)
        #             labelList.append(labelIndex)
        #     # 训练集和测试集比例为9：1
        #     train_x, test_x, train_y, test_y = train_test_split(nameList, labelList, test_size=0.1, stratify=labelList,
        #                                                         random_state=5)  # 5
        #     # 在训练集中的有标签数据和无标签数据比例为2：7
        #     unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(train_x, train_y, test_size=0.222,
        #                                                                       stratify=train_y, random_state=5)  # 5
        #     # 有标签数据的标签和名字的Tuplelist
        #     Label_Name_Labeled_train_tuple_list = self.__get_Y_X_tuple_list(labeled_y, labeled_x)
        #     # 无标签数据的标签和名字Tuplelist
        #     Label_Name_Unlabeled_train_tuple_list = self.__get_Y_X_tuple_list(unlabeled_y, unlabeled_x)
        #     # 测试集数据的标签和名字Tuplelist
        #     Label_Name_test_tuple_list = self.__get_Y_X_tuple_list(test_y, test_x)
        #     return Label_Name_Labeled_train_tuple_list, Label_Name_Unlabeled_train_tuple_list, Label_Name_test_tuple_list

    # def getLabelAndUnlabelAndTestTupleList(self, featureDirPath):
    #     """
    #     获取有标签集合，无标签集合，测试集合
    #     :param featureDirPath:特征文件夹地址
    #     :return:有标签集合，无标签集合，测试集合
    #     """
    #     labelAndFeatureTrainTupleList, labelAndFeatureTestTupleList = self.__getFeatureAndLabelTuple(featureDirPath)
    #     Y_list, X_list = self.__get_Y_and_X_list_from_tuple(labelAndFeatureTrainTupleList)
    #     unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(X_list, Y_list, test_size=0.222,
    #                                                                       stratify=Y_list)
    #     labelAndFeatureLabeledTupleList = self.__get_Y_X_tuple_list(labeled_y, labeled_x)
    #     labelAndFeatureUnlabeledTupleList = self.__get_Y_X_tuple_list(unlabeled_y, unlabeled_x)
    #     return labelAndFeatureLabeledTupleList, labelAndFeatureUnlabeledTupleList, labelAndFeatureTestTupleList

    def loadData(self, featureDirPath, featureDir, tupleList):
        """
        获取标签和特征tuplelist
        :param featureDirPath:特征文件夹地址
        :param tupleList:标签和特征名字的tuplelist
        :return:获取标签和特征tuplelist
        """
        # 标签和特征元组list
        labelAndFeatureAndNameTupleList = []
        for Label_Name_tuple in tupleList:
            featurePath = featureDirPath + Label_Name_tuple[1].split("_v")[0] + featureDir + Label_Name_tuple[1]
            # 读取特征和标签组成元组
            labelAndFeatureAndNameTuple = (Label_Name_tuple[0], np.loadtxt(featurePath), Label_Name_tuple[1])
            labelAndFeatureAndNameTupleList.append(labelAndFeatureAndNameTuple)
        return labelAndFeatureAndNameTupleList

        # labelAndFeatureTrainTupleList, labelAndFeatureTestTupleList = self.__getFeatureAndLabelTuple(featureDirPath)
        # Y_list, X_list = self.__get_Y_and_X_list_from_tuple(labelAndFeatureTrainTupleList)
        # unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(X_list, Y_list, test_size=0.222,
        #                                                                   stratify=Y_list)
        # labelAndFeatureLabeledTupleList = self.__get_Y_X_tuple_list(labeled_y, labeled_x)
        # labelAndFeatureUnlabeledTupleList = self.__get_Y_X_tuple_list(unlabeled_y, unlabeled_x)
        # return labelAndFeatureLabeledTupleList, labelAndFeatureUnlabeledTupleList, labelAndFeatureTestTupleList

    def getBootstrapSample(self, tupleList, random_state):
        """
        把有标签数据集进行Bootstrap取样
        :param tupleList:有标签数据集的标签和名字tupleList
        :param random_state:随机种子
        :return:取样后的标签和名字tupleList
        """

        Y_list, X_list = self.__get_Y_and_X_list_from_tuple(tupleList)
        # 在训练样本集上取百分之70
        sampled_x, unsampled_x, sampled_y, unsampled_y = train_test_split(X_list, Y_list, test_size=0.3,
                                                                          stratify=Y_list, random_state=random_state)
        return self.__get_Y_X_tuple_list(sampled_y, sampled_x)

    # def __getFeatureAndLabelTuple(self, featureDirPath):
    #     """
    #     获取特征和标签
    #     :param featureDirPath:存放feature的文件夹路径
    #     :return:
    #     """
    #     # 训练集中的标签和特征元组list
    #     labelAndFeatureTrainTupleList = []
    #     # 测试集中的标签和特征元组list
    #     labelAndFeatureTestTupleList = []
    #     for Label_Name_train_tuple in self.__Label_Name_train_tuple_list:
    #         featureTrainPath = featureDirPath + Label_Name_train_tuple[1]
    #         # 读取特征和标签组成元组
    #         labelAndFeatureTrainTuple = (Label_Name_train_tuple[0], [np.loadtxt(featureTrainPath)])
    #         labelAndFeatureTrainTupleList.append(labelAndFeatureTrainTuple)
    #     for Label_Name_test_tuple in self.__Label_Name_test_tuple_list:
    #         featureTestPath = featureDirPath + Label_Name_test_tuple[1]
    #         # 读取特征和标签组成元组
    #         labelAndFeatureTestTuple = (Label_Name_test_tuple[0], [np.loadtxt(featureTestPath)])
    #         labelAndFeatureTestTupleList.append(labelAndFeatureTestTuple)
    #     return labelAndFeatureTrainTupleList, labelAndFeatureTestTupleList

    def __get_Y_X_tuple_list(self, Y_list, X_list):
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
            return Y_X_tuple_list

    def __get_Y_and_X_list_from_tuple(self, Y_X_tuple_list):
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
