# -*- coding: utf-8 -*-
from myFirstPoint.DataPreparation import DataPreparation
import myFirstPoint.MFSSEL_Utilities as Utilities
from sklearn.svm import SVC


def MFSSEL(featureNameDirPath):
    _122FeatureDir = "/home/sunbite/features/_122videoFeature/"
    _81FeatureDir = "/home/sunbite/features/_81videoFeature/"
    _30_5_6FeatureDir = "/home/sunbite/features/_30_5_6videoFeature/"
    _81_5_6FeatureDir = "/home/sunbite/features/_81_5_6videoFeature/"
    _30FeatureDir = "/home/sunbite/features/_30videoFeature/"
    # SVM保存准确率
    accuracy_svm_list = []

    # 初始化dataPreparation对象
    dataPreparation = DataPreparation()
    # 获取有标签训练集，无标签训练集，测试集的标签和名字tuplelist
    label_name_labeled_train_tuple_list, label_name_unlabeled_train_tuple_list, label_name_test_tuple_list = dataPreparation.getLabelAndNameTupleList(
        featureNameDirPath)
    # 取样后的有标签训练集的标签和名字tuplelist
    bootstrapped_Labeled_train_tuple_list_1 = dataPreparation.getBootstrapSample(label_name_labeled_train_tuple_list, 1)
    bootstrapped_Labeled_train_tuple_list_2 = dataPreparation.getBootstrapSample(label_name_labeled_train_tuple_list,
                                                                                 40)

    # 获取122维有标签训练集
    _122_labeled_train_tuple_list = dataPreparation.loadData(_122FeatureDir, bootstrapped_Labeled_train_tuple_list_1)
    _122_labeled_train_Y, _122_labeled_train_X, _122_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _122_labeled_train_tuple_list)
    # 获取122维无标签训练集
    _122_unlabeled_tuple_list = dataPreparation.loadData(_122FeatureDir, label_name_unlabeled_train_tuple_list)
    _122_unlabeled_Y, _122_unlabeled_X, _122_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _122_unlabeled_tuple_list)
    # 获取122维测试集
    _122_test_tuple_list = dataPreparation.loadData(_122FeatureDir, label_name_test_tuple_list)
    _122_test_Y, _122_test_X, _122_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_122_test_tuple_list)

    # 获取81维有标签训练集
    _81_labeled_train_tuple_list = dataPreparation.loadData(_81FeatureDir, bootstrapped_Labeled_train_tuple_list_1)
    _81_labeled_train_Y, _81_labeled_train_X, _81_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _81_labeled_train_tuple_list)
    # 获取81维无标签训练集
    _81_unlabeled_tuple_list = dataPreparation.loadData(_81FeatureDir, label_name_unlabeled_train_tuple_list)
    _81_unlabeled_Y, _81_unlabeled_X, _81_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _81_unlabeled_tuple_list)
    # 获取81维测试集
    _81_test_tuple_list = dataPreparation.loadData(_81FeatureDir, label_name_test_tuple_list)
    _81_test_Y, _81_test_X, _81_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_81_test_tuple_list)

    # 获取30_5_6维有标签训练集
    _30_5_6_labeled_train_tuple_list = dataPreparation.loadData(_30_5_6FeatureDir,
                                                                bootstrapped_Labeled_train_tuple_list_1)
    _30_5_6_labeled_train_Y, _30_5_6_labeled_train_X, _30_5_6_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _30_5_6_labeled_train_tuple_list)
    # 获取30_5_6维无标签训练集
    _30_5_6_unlabeled_tuple_list = dataPreparation.loadData(_30_5_6FeatureDir, label_name_unlabeled_train_tuple_list)
    _30_5_6_unlabeled_Y, _30_5_6_unlabeled_X, _30_5_6_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _30_5_6_unlabeled_tuple_list)
    # 获取30_5_6维测试集
    _30_5_6_test_tuple_list = dataPreparation.loadData(_30_5_6FeatureDir, label_name_test_tuple_list)
    _30_5_6_test_Y, _30_5_6_test_X, _30_5_6_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_30_5_6_test_tuple_list)

    # 获取81_5_6维有标签训练集
    _81_5_6_labeled_train_tuple_list = dataPreparation.loadData(_81_5_6FeatureDir,
                                                                bootstrapped_Labeled_train_tuple_list_1)
    _81_5_6_labeled_train_Y, _81_5_6_labeled_train_X, _81_5_6_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _81_5_6_labeled_train_tuple_list)
    # 获取81_5_6维无标签训练集
    _81_5_6_unlabeled_tuple_list = dataPreparation.loadData(_81_5_6FeatureDir, label_name_unlabeled_train_tuple_list)
    _81_5_6_unlabeled_Y, _81_5_6_unlabeled_X, _81_5_6_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _81_5_6_unlabeled_tuple_list)
    # 获取81_5_6维测试集
    _81_5_6_test_tuple_list = dataPreparation.loadData(_81_5_6FeatureDir, label_name_test_tuple_list)
    _81_5_6_test_Y, _81_5_6_test_X, _81_5_6_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_81_5_6_test_tuple_list)

    # 获取30维有标签训练集
    _30_labeled_train_tuple_list = dataPreparation.loadData(_30FeatureDir, bootstrapped_Labeled_train_tuple_list_1)
    _30_labeled_train_Y, _30_labeled_train_X, _30_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _30_labeled_train_tuple_list)
    # 获取30维无标签训练集
    _30_unlabeled_tuple_list = dataPreparation.loadData(_30FeatureDir, label_name_unlabeled_train_tuple_list)
    _30_unlabeled_Y, _30_unlabeled_X, _30_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
        _30_unlabeled_tuple_list)
    # 获取30维测试集
    _30_test_tuple_list = dataPreparation.loadData(_30FeatureDir, label_name_test_tuple_list)
    _30_test_Y, _30_test_X, _30_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_30_test_tuple_list)

    # # svm计算准确率
    # _122svc_1 = SVC(C=16, kernel='rbf', gamma=5, probability=True)  # gamma=7
    # # 训练
    # _122svc_1.fit(_122_labeled_train_X, _122_labeled_train_Y)

    # svm计算准确率
    _81svc_1 = SVC(C=16, kernel='rbf', gamma=5, probability=True)  # gamma=7
    # 训练
    _81svc_1.fit(_81_labeled_train_X, _81_labeled_train_Y)

    # # svm计算准确率
    # _30_5_6svc_1 = SVC(C=16, kernel='rbf', gamma=5, probability=True)  # gamma=7
    # # 训练
    # _30_5_6svc_1.fit(_30_5_6_labeled_train_X, _30_5_6_labeled_train_Y)
    #
    # # svm计算准确率
    # _81_5_6svc_1 = SVC(C=16, kernel='rbf', gamma=5, probability=True)  # gamma=7
    # # 训练
    # _81_5_6svc_1.fit(_81_5_6_labeled_train_X, _81_5_6_labeled_train_Y)
    #
    # # svm计算准确率
    # _30svc_1 = SVC(C=16, kernel='rbf', gamma=5, probability=True)  # gamma=7
    # # 训练
    # _30svc_1.fit(_30_labeled_train_X, _30_labeled_train_Y)

    _81_svc_1_probility = _81svc_1.predict_proba(_81_unlabeled_X)

    # 获得准确率
    accuracy_svm = _81svc_1.score(_81_test_X, _81_test_Y)

    accuracy_svm_list.append(accuracy_svm * 100)


if __name__ == '__main__':
    MFSSEL("/home/sunbite/features/_122videoFeature/")
