# -*- coding: utf-8 -*-

import Co_KNN_SVM_Utilities as utilities
import svmutil
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def Co_KNN_SVM(train_Y, train_X, test_Y, test_X, savepath=None):
    # 每次迭代，添加到对方分类器训练集的样本数
    temp_num_svm = 44
    temp_num_knn = 44

    # 迭代次数
    loop_num = 6

    # knn中的K
    K = 4

    # KNN和SVM用来测试的样本及测试的标签（不变）
    fixed_test_X = test_X.copy()
    fixed_test_Y = test_Y.copy()

    # KNN保存准确率
    accuracy_knn_list = []
    # SVM保存准确率
    accuracy_svm_list = []

    # knn训练标签和训练集特征组成的元组list
    train_knn_Y_X_tuple_list = utilities.get_Y_X_tuple_list(train_Y.copy(), train_X.copy())
    # knn测试标签和测试集特征组成的元组list
    test_knn_Y_X_tuple_list = utilities.get_Y_X_tuple_list(test_Y.copy(), test_X.copy())
    # svm训练标签和训练集特征组成的元组list
    train_svm_Y_X_tuple_list = utilities.get_Y_X_tuple_list(train_Y.copy(), train_X.copy())
    # svm测试标签和测试集特征组成的元组list
    test_svm_Y_X_tuple_list = utilities.get_Y_X_tuple_list(test_Y.copy(), test_X.copy())
    # 协同训练
    for h in range(1, loop_num + 1):
        print(len(train_knn_Y_X_tuple_list))
        print(len(test_knn_Y_X_tuple_list))
        print(len(train_svm_Y_X_tuple_list))
        print(len(test_svm_Y_X_tuple_list))
        # 得到svm的训练集标签和训练集的特征
        train_Y_svm_from_tuple, train_X_svm_from_tuple = utilities.get_Y_and_X_list_from_tuple(
            train_svm_Y_X_tuple_list.copy())
        # 得到svm的测试集标签和测试集的特征
        test_Y_svm_from_tuple, test_X_svm_from_tuple = utilities.get_Y_and_X_list_from_tuple(
            test_svm_Y_X_tuple_list.copy())
        # 得到knn的训练集标签和训练集的特征
        train_Y_knn_from_tuple, train_X_knn_from_tuple = utilities.get_Y_and_X_list_from_tuple(train_knn_Y_X_tuple_list)
        # 得到knn的测试集标签和测试集的特征
        test_Y_knn_from_tuple, test_X_knn_from_tuple = utilities.get_Y_and_X_list_from_tuple(test_knn_Y_X_tuple_list)

        # KNN计算准确率
        knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
        # 训练
        knn.fit(train_X_knn_from_tuple, train_Y_knn_from_tuple)
        # 获得准确率
        accuracy_knn = knn.score(fixed_test_X, fixed_test_Y)
        # accuracy_knn = knn.score(test_X_knn_from_tuple, test_Y_knn_from_tuple)
        accuracy_knn_list.append(accuracy_knn * 100)

        print("预测结果（KNN）")
        print(h)
        print(accuracy_knn)

        # svm计算准确率
        svc = SVC(C=15, kernel='rbf', degree=3, gamma=2, probability=True)
        # 训练
        svc.fit(train_X_svm_from_tuple, train_Y_svm_from_tuple)
        # 获得准确率
        accuracy_svm = svc.score(fixed_test_X, fixed_test_Y)
        # accuracy_svm = svc.score(test_X_svm_from_tuple, test_Y_svm_from_tuple)
        accuracy_svm_list.append(accuracy_svm * 100)

        print("预测结果（SVM）")
        print(h)
        print(accuracy_svm)

        if h == loop_num:
            break

        # KNN和SVM半监督训练过程
        # ---------------------------------KNN测试样本预测和置信度计算过程 ----------------------------------
        # 根据模型，预测样本
        # 获得预测可能性
        probility_knn = knn.predict_proba(test_X_knn_from_tuple)
        # knn的置信list
        confidence_knn_list = []
        for i in range(0, probility_knn.shape[0]):
            probility_knn_temp = probility_knn[i]
            confidence_knn_list.append(utilities.get_confidence_knn(probility_knn_temp.copy()))

        # 获得预测标签
        predict_Y_knn = knn.predict(test_X_knn_from_tuple)

        # ---------------------------------SVM测试样本预测和置信度计算过程 ----------------------------------
        # 根据模型，预测样本
        # 获得预测可能性
        probility_svm = svc.predict_proba(test_X_svm_from_tuple)

        # svm的置信list
        confidence_svm_list = []
        for i in range(0, probility_svm.shape[0]):
            probility_svm_temp = probility_svm[i]
            confidence_svm_list.append(utilities.get_confidence_svm(probility_svm_temp.copy()))

        # 获得预测标签
        predict_Y_svm = svc.predict(test_X_svm_from_tuple)

        # KNN和SVM伪标签添加过程
        # ---------------------------------------KNN---------------------------------------------
        index_svm_label_high_confidence = utilities.get_confidence_svm_index(confidence_svm_list.copy(),
                                                                             predict_Y_svm.copy(),
                                                                             predict_Y_knn.copy(),
                                                                             temp_num_svm)

        temp_test_X_svm = []
        temp_test_Y_svm = []

        for i in index_svm_label_high_confidence:
            temp_test_X_svm.append(test_X_svm_from_tuple[i])
            temp_test_Y_svm.append(predict_Y_svm[i])

        temp_test_svm_Y_X_tuple_list = utilities.get_Y_X_tuple_list(temp_test_Y_svm.copy(), temp_test_X_svm.copy())
        # 把svm的置信度较高的样本加入到knn的训练集中
        train_knn_Y_X_tuple_list.extend(temp_test_svm_Y_X_tuple_list)

        # 获取新的测试样本
        # index_all_test_svm_Y_X_tuple_list = np.arange(0, len(test_svm_Y_X_tuple_list))
        # diff_index_test_svm_Y_X_tuple_list = np.setdiff1d(index_all_test_svm_Y_X_tuple_list,
        #                                                   np.array(index_svm_label_high_confidence))
        # diff_test_svm_Y_X_tuple_list = []
        # for i in diff_index_test_svm_Y_X_tuple_list:
        #     diff_test_svm_Y_X_tuple_list.append(test_svm_Y_X_tuple_list[i])
        # test_svm_Y_X_tuple_list = diff_test_svm_Y_X_tuple_list
        for i in index_svm_label_high_confidence:
            print("test_svm_Y_X_tuple_list的长度：")
            print(len(test_svm_Y_X_tuple_list))
            print("i：")
            print(i)
            test_svm_Y_X_tuple_list.pop(i)

        # ---------------------------------------SVM---------------------------------------------
        index_knn_label_high_confidence = utilities.get_confidence_knn_index(confidence_knn_list.copy(),
                                                                             predict_Y_svm.copy(),
                                                                             predict_Y_knn.copy(),
                                                                             temp_num_knn)

        temp_test_X_knn = []
        temp_test_Y_knn = []

        for i in index_knn_label_high_confidence:
            temp_test_X_knn.append(test_X_knn_from_tuple[i])
            temp_test_Y_knn.append(predict_Y_knn[i])

        temp_test_knn_Y_X_tuple_list = utilities.get_Y_X_tuple_list(temp_test_Y_knn.copy(), temp_test_X_knn.copy())
        # 把knn的置信度较高的样本加入到svm的训练集中
        train_svm_Y_X_tuple_list.extend(temp_test_knn_Y_X_tuple_list)
        # 获取新的测试样本
        # index_all_test_knn_Y_X_tuple_list = np.arange(0, len(test_knn_Y_X_tuple_list))
        # diff_index_test_knn_Y_X_tuple_list = np.setdiff1d(index_all_test_knn_Y_X_tuple_list,
        #                                                  np.array(index_knn_label_high_confidence))
        # test_knn_Y_X_tuple_list
        # diff_test_knn_Y_X_tuple_list = []
        # for i in diff_index_test_knn_Y_X_tuple_list:
        #     diff_test_knn_Y_X_tuple_list.append(test_knn_Y_X_tuple_list[i])
        # test_knn_Y_X_tuple_list = diff_test_knn_Y_X_tuple_list
        for i in index_knn_label_high_confidence:
            test_knn_Y_X_tuple_list.pop(i)

    print("KNN的准确率：")
    print(accuracy_knn_list)
    print("SVM的准确率：")
    print(accuracy_svm_list)
