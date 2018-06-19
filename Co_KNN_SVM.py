# -*- coding: utf-8 -*-

import Co_KNN_SVM_Utilities as utilities
import svmutil
import numpy as np


def Co_KNN_SVM(train_Y, train_X, test_Y, test_X, savepath=None):
    # 每次迭代，添加到对方分类器训练集的样本数
    temp_num_svm = 660
    temp_num_knn = 660
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
    # KNN保存准确率
    accuracy_knn_list = []
    # SVM保存准确率
    accuracy_svm_list = []
    # Co_KNN_SVM保存准确率
    accuracy_Co_KNN_SVM_list = []

    # 协同训练
    for h in range(1, loop_num + 1):
        print(len(train_X_knn))
        print(len(test_X_knn))
        print(len(train_X_svm))
        print(len(test_X_svm))
        # KNN计算准确率
        predict_Y_knn, accuracy_knn, probility_knn = utilities.knn_sklearn(train_X_knn.copy(), train_Y_knn.copy(),
                                                                           test_X_knn.copy(), test_Y_knn.copy(),
                                                                           K)
        # knn的置信list
        confidence_knn_list = []
        for i in range(0, probility_knn.shape[0]):
            probility_knn_temp = probility_knn[i]
            confidence_knn_list.append(utilities.get_confidence_knn(probility_knn_temp.copy()))
        accuracy_knn_list.append(accuracy_knn * 100)

        train_X_svm_for_libsvm = utilities.get_feature_for_libsvm(train_X_svm.copy())
        # 训练svm模型
        model = svmutil.svm_train(train_Y_svm.copy(), train_X_svm_for_libsvm.copy(), '-s 0 -t 2 -c 15 -g 2 -b 1 -q')
        test_X_svm_for_libsvm = utilities.get_feature_for_libsvm(test_X_svm.copy())
        # SVM计算准确率
        predict_Y_svm, accuracy_svm, probility_svm = svmutil.svm_predict(test_Y_svm.copy(),
                                                                         test_X_svm_for_libsvm.copy(),
                                                                         model, '-b 1')
        accuracy_svm_list.append(accuracy_svm[0])
        accuracy_Co_KNN_SVM_list.append(accuracy_svm[0])

        # svm的置信度list
        confidence_svm_list = []
        for i in range(0, len(test_X_svm)):
            probility_svm_temp = probility_svm[i]
            confidence_svm_list.append(utilities.get_confidence_svm(probility_svm_temp.copy()))
        print(h)
        print(loop_num)
        if h == loop_num:
            # svmutil.svm_save_model(savepath, model)
            break
        # KNN和SVM半监督训练过程
        # KNN和SVM伪标签添加过程
        # ---------------------------------------KNN---------------------------------------------
        index_svm_label_high_confidence = utilities.get_confidence_svm_index(confidence_svm_list.copy(),
                                                                             predict_Y_svm.copy(), predict_Y_knn.copy(),
                                                                             temp_num_svm)
        temp_test_X_svm = []
        temp_test_Y_svm = []

        for i in index_svm_label_high_confidence:
            temp_test_X_svm.append(test_X_svm[i])
            temp_test_Y_svm.append(predict_Y_svm[i])
        # 把svm的置信度较高的样本加入到knn的训练集中
        # train_X_knn.extend(temp_test_X_svm)
        # train_Y_knn.extend(temp_test_Y_svm)
        train_X_knn.extend(temp_test_X_svm)
        train_Y_knn.extend(temp_test_Y_svm)
        # 获取新的svm测试样本
        index_all_test_X_svm = np.arange(0, len(test_X_svm))
        diff_index_test_X_svm = np.setdiff1d(index_all_test_X_svm, np.array(index_svm_label_high_confidence))
        diff_test_X_svm = []
        diff_test_Y_svm = []
        for i in diff_index_test_X_svm:
            diff_test_X_svm.append(test_X_svm[i])
            diff_test_Y_svm.append(test_Y_svm[i])
        test_X_svm = diff_test_X_svm
        test_Y_svm = diff_test_Y_svm

        # ---------------------------------------SVM---------------------------------------------
        index_knn_label_high_confidence = utilities.get_confidence_knn_index(confidence_knn_list.copy(),
                                                                             predict_Y_svm.copy(), predict_Y_knn.copy(),
                                                                             temp_num_knn)

        temp_test_X_knn = []
        temp_test_Y_knn = []

        for i in index_knn_label_high_confidence:
            temp_test_X_knn.append(test_X_knn[i])
            temp_test_Y_knn.append(predict_Y_knn[i])
        # 把knn的置信度较高的样本加入到svm的训练集中
        # train_X_svm.extend(temp_test_X_knn)
        # train_Y_svm.extend(temp_test_Y_knn)
        train_X_svm.extend(temp_test_X_knn)
        train_Y_svm.extend(temp_test_Y_knn)
        # 获取新的测试样本
        index_all_test_X_knn = np.arange(0, len(test_X_knn))
        diff_index_test_X_knn = np.setdiff1d(index_all_test_X_knn, np.array(index_knn_label_high_confidence))
        diff_test_X_knn = []
        diff_test_Y_knn = []
        for i in diff_index_test_X_knn:
            diff_test_X_knn.append(test_X_knn[i])
            diff_test_Y_knn.append(test_Y_knn[i])
        test_X_knn = diff_test_X_knn
        test_Y_knn = diff_test_Y_knn

    print("Co_KNN_SVM的准确率：")
    print(accuracy_knn_list)
    print(accuracy_Co_KNN_SVM_list)


if __name__ == '__main__':
    # KNN和SVM训练样本
    train_Y, train_X = svmutil.svm_read_problem("/home/sunbite/train")
    test_Y, test_X = svmutil.svm_read_problem("/home/sunbite/test")
    Co_KNN_SVM(train_Y, utilities.get_feature_from_libsvm(train_X), test_Y, utilities.get_feature_from_libsvm(test_X))
