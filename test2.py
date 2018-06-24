# -*- coding: utf-8 -*-

import Co_KNN_SVM_Utilities as utilities
import svmutil
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn import datasets
from sklearn.svm import SVC


def Co_KNN_SVM(train_Y, train_X, test_Y, test_X, savepath=None):
    # 每次迭代，添加到对方分类器训练集的样本数
    temp_num_svm = 220
    temp_num_knn = 220
    # 迭代次数
    loop_num = 6
    # knn中的K
    K = 4
    # knn训练集
    train_X_knn = train_X.copy()
    # knn训练标签
    train_Y_knn = train_Y.copy()
    # svm训练集
    train_X_svm = train_X.copy()
    # svm训练标签
    train_Y_svm = train_Y.copy()
    # knn测试集
    test_X_knn = test_X.copy()
    # knn测试标签
    test_Y_knn = test_Y.copy()
    # svm测试集
    test_X_svm = test_X.copy()
    # svm测试标签
    test_Y_svm = test_Y.copy()

    # knn测试集
    fixed_test_X = test_X.copy()
    # knn测试标签
    fixed_test_Y = test_Y.copy()
    # KNN保存准确率
    accuracy_knn_list = []
    # SVM保存准确率
    accuracy_svm_list = []
    # Co_KNN_SVM保存准确率
    accuracy_Co_KNN_SVM_list = []
    train_Y_X_tuple_list = utilities.get_Y_X_tuple_list(train_Y_knn.copy(), train_X_knn.copy())
    test_Y_X_tuple_list = utilities.get_Y_X_tuple_list(test_Y_knn.copy(), test_X_knn.copy())
    # 协同训练
    for h in range(1, loop_num + 1):
        print(len(train_Y_X_tuple_list))
        print(len(test_Y_X_tuple_list))
        train_Y_knn_from_tuple, train_X_knn_from_tuple = utilities.get_Y_and_X_list_from_tuple(train_Y_X_tuple_list)
        test_Y_knn_from_tuple, test_X_knn_from_tuple = utilities.get_Y_and_X_list_from_tuple(test_Y_X_tuple_list)

        # KNN计算准确率
        knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
        # 训练
        knn.fit(train_X_knn_from_tuple, train_Y_knn_from_tuple)
        # 获得准确率
        # accuracy_knn = knn.score(fixed_test_X, fixed_test_Y)
        accuracy_knn = knn.score(test_X_knn_from_tuple, test_Y_knn_from_tuple)
        accuracy_knn_list.append(accuracy_knn * 100)
        print(h)
        print(loop_num)

        # # svm计算准确率
        # svc = SVC(C=15, kernel='rbf', degree=3, gamma=2, probability=True)
        # # 训练
        # svc.fit(train_X_knn_from_tuple, train_Y_knn_from_tuple)
        # # 获得准确率
        # # accuracy_svm = svc.score(fixed_test_X, fixed_test_Y)
        # accuracy_svm = svc.score(test_X_knn_from_tuple, test_Y_knn_from_tuple)
        # accuracy_svm_list.append(accuracy_svm * 100)
        #
        # print("预测结果（SVM）")
        # print(h)
        # print(accuracy_svm)
        if h == loop_num:
            # svmutil.svm_save_model(savepath, model)
            break
        # temp_test_Y_X_tuple_list = []
        # for i in range(0, temp_num_svm):
        #     temp_test_Y_X_tuple_list.append(test_Y_X_tuple_list[i])
        #
        # train_Y_X_tuple_list.extend(temp_test_Y_X_tuple_list)
        # diff_test_Y_X_tuple_list = []
        # for i in range(temp_num_svm, len(test_Y_X_tuple_list)):
        #     diff_test_Y_X_tuple_list.append(test_Y_X_tuple_list[i])
        # test_Y_X_tuple_list = diff_test_Y_X_tuple_list
        # KNN计算准确率
        probility_knn = knn.predict_proba(test_X_knn_from_tuple)
        # knn的置信list
        confidence_knn_list = []
        for i in range(0, probility_knn.shape[0]):
            probility_knn_temp = probility_knn[i]
            confidence_knn_list.append(utilities.get_confidence_knn(probility_knn_temp.copy()))

        # 获得预测标签
        predict_Y_knn = knn.predict(test_X_knn_from_tuple)
        index_knn_label_high_confidence = utilities.get_confidence_knn_index(confidence_knn_list.copy(),
                                                                             test_Y_knn_from_tuple.copy(),
                                                                             predict_Y_knn.copy(),

                                                                             temp_num_svm)

        # index_knn_label_high_confidence = random.sample(range(len(test_Y_knn_from_tuple)),220)
        #index_knn_label_high_confidence = np.asarray(index_knn_label_high_confidence)
        # index_knn_label_high_confidence = np.arange(0, 220)
        temp_test_X_knn = []
        temp_test_Y_knn = []

        for i in index_knn_label_high_confidence:
            temp_test_X_knn.append(test_X_knn_from_tuple[i])
            # temp_test_Y_knn.append(predict_Y_knn[i])
            temp_test_Y_knn.append(test_Y_knn_from_tuple[i])

        temp_test_Y_X_tuple_list = utilities.get_Y_X_tuple_list(temp_test_Y_knn.copy(), temp_test_X_knn.copy())

        train_Y_X_tuple_list.extend(temp_test_Y_X_tuple_list)
        #index_knn_label_high_confidence = range(len(temp_test_Y_X_tuple_list))
        #for i in index_knn_label_high_confidence:
        #index_knn_label_high_confidence=[0]
        test_Y_X_tuple_list = [i for j, i in enumerate(test_Y_X_tuple_list) if j not in index_knn_label_high_confidence]
        #test_Y_X_tuple_list.pop(0)

        # index_all_test_Y_X_tuple_list = range(len(test_Y_X_tuple_list))
        # diff_index_test_Y_X_tuple_list = np.setdiff1d(index_all_test_Y_X_tuple_list,
        #                                               index_knn_label_high_confidence)
        # diff_test_Y_X_tuple_list = []
        # for i in diff_index_test_Y_X_tuple_list:
        #     diff_test_Y_X_tuple_list.append(test_Y_X_tuple_list[i])
        # test_Y_X_tuple_list = diff_test_Y_X_tuple_list

    print("SVM的准确率：")
    print(accuracy_svm_list)
    print("KNN的准确率：")
    print(accuracy_knn_list)


if __name__ == '__main__':
    x, y = datasets.load_svmlight_file("/home/sunbite/dataset/dataset")
    train_x, test_x, train_y, test_y = train_test_split(x.todense().tolist(), y, test_size=0.8, random_state=42)

    Co_KNN_SVM(train_y, train_x, test_y, test_x)
