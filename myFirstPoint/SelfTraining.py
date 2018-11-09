# -*- coding: utf-8 -*-
from myFirstPoint.DataPreparation import DataPreparation
import myFirstPoint.MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
from sklearn.externals import joblib
import os


def selfTraining(featureNameDirPath, savePath=None, trainAndTestFlag="train"):
    model_max_hog = None
    model_max_81 = None
    model_max_30 = None
    accuracy_max = 0
    accuracy_max_from_single = 0
    featureDir = "/home/sunbite/MFSSEL/features_new_1/"
    train_81FeatureDir = "/train/_81videoFeature/"
    train_30FeatureDir = "/train/_30videoFeature/"
    train_hogFeatureDir = "/train/hogvideoFeature/"
    test_81FeatureDir = "/test/_81videoFeature/"
    test_30FeatureDir = "/test/_30videoFeature/"
    test_hogFeatureDir = "/test/hogvideoFeature/"

    # 准确率
    accuracy_list = []
    # 整体类别
    whole_class = 11
    # 选取置信度前topk个样本
    topk = 30
    # 迭代次数
    loop_num = 10000
    # 有标签数据集大小list
    label_data_num_list = []

    hog_real_Y = []
    hog_predict_Y = []

    # 初始化dataPreparation对象
    dataPreparation = DataPreparation()
    # 获取有标签训练集，无标签训练集，测试集的标签和名字tuplelist
    label_name_labeled_train_tuple_list, label_name_unlabeled_train_tuple_list, label_name_test_tuple_list = dataPreparation.getLabelAndNameTupleList(
        featureNameDirPath)

    # # 取样后的有标签训练集的标签和名字tuplelist
    # bootstrapped_Labeled_train_tuple_list_1 = dataPreparation.getBootstrapSample(label_name_labeled_train_tuple_list, 1)
    # bootstrapped_Labeled_train_tuple_list_2 = dataPreparation.getBootstrapSample(label_name_labeled_train_tuple_list,
    #                                                                              40)
    # 获取hog维有标签训练集
    hog_labeled_train_tuple_list = dataPreparation.loadData(featureDir, train_hogFeatureDir,
                                                            label_name_labeled_train_tuple_list)
    hog_labeled_train_Y, hog_labeled_train_X, hog_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        hog_labeled_train_tuple_list)
    # 获取hog维无标签训练集
    hog_unlabeled_tuple_list = dataPreparation.loadData(featureDir, train_hogFeatureDir,
                                                        label_name_unlabeled_train_tuple_list)
    hog_unlabeled_Y, hog_unlabeled_X, hog_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        hog_unlabeled_tuple_list)
    # 获取hog维测试集
    hog_test_tuple_list = dataPreparation.loadData(featureDir, test_hogFeatureDir, label_name_test_tuple_list)
    hog_test_Y, hog_test_X, hog_test_Name = Utilities.get_Y_X_Name_list_from_tuple(hog_test_tuple_list)

    for h in range(1, loop_num + 1):
        print("有标签数据集大小：")
        label_data_num_list.append(len(hog_labeled_train_Y))
        print(label_data_num_list)
        #print(len(hog_unlabeled_Y))
        # hog维svm训练
        hog_svc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # c:4 gamma=2
        hog_svc_1.fit(hog_labeled_train_X, hog_labeled_train_Y)

        # 获得准确率
        hog_accuracy = hog_svc_1.score(hog_test_X, hog_test_Y)
        accuracy_list.append(hog_accuracy * 100)
        print("accuracy_list:")
        print(accuracy_list)

        if hog_accuracy > accuracy_max:
            accuracy_max = hog_accuracy
            model_max_hog = hog_svc_1

        if len(hog_unlabeled_X) == 0:
            break
        if h == loop_num:
            break

        # hog特征下的无标签数据集的所对应的各个类别的概率
        hog_svc_1_probility = hog_svc_1.predict_proba(hog_unlabeled_X)
        # hog特征下无标签数据的预测标签
        hog_svc_1_predict_Y = hog_svc_1.predict(hog_unlabeled_X)
        ind_confidence_list, ind_confidence_list_backup = Utilities.get_confidence_selfTraining_index(
            hog_svc_1_probility,
            topk)

        # voted_index_predict_Y_list = Utilities.vote(predict_Y_list_1, unlabeled_Y_list_1, whole_class, topk)
        #
        # voted_Y_list, voted_index_list = Utilities.get_voted_confidence(probility_list_1,
        #                                                                 voted_index_predict_Y_list[0],
        #                                                                 voted_index_predict_Y_list[1], whole_class,
        #                                                                 topk)


        if len(ind_confidence_list) == 0:
            for i in ind_confidence_list_backup:
                hog_real_Y.append(hog_unlabeled_Y[i])
                hog_predict_Y.append(hog_svc_1_predict_Y[i])
                hog_labeled_train_X.append(hog_unlabeled_X[i])
                hog_labeled_train_Y.append(hog_unlabeled_Y[i])
            hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in ind_confidence_list_backup]
            hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in ind_confidence_list_backup]
        else:
            for i in ind_confidence_list:
                hog_real_Y.append(hog_unlabeled_Y[i])
                hog_predict_Y.append(hog_svc_1_predict_Y[i])
                hog_labeled_train_X.append(hog_unlabeled_X[i])
                hog_labeled_train_Y.append(hog_svc_1_predict_Y[i])

            hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in ind_confidence_list]
            hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in ind_confidence_list]


    print(accuracy_max * 100)
    if model_max_hog is not None:
        print("正在保存selfTraining.model...")
        joblib.dump(model_max_hog, savePath + "selfTraining.model")
        print("保存selfTraining.model完毕。")


if __name__ == '__main__':
    selfTraining("/home/sunbite/MFSSEL/features_new_1/", "/home/sunbite/MFSSEL/model/")
