# -*- coding: utf-8 -*-

import scipy.io
import Co_KNN_SVM_Utilities as utilities
import svmutil
import numpy as np
import matplotlib.pyplot as plt

def co_knn_svm(trainLabels, trainImages, testLabels, testImages):
    # 获取训练集和测试集
    # mapping = scipy.io.loadmat('pingjun_jingchenyong2_8.mat')
    # 每次迭代，选择伪标签
    temp_num_SVM = 300
    temp_num = 300
    loop_num = 6
    K = 5  # can be any other value

    # KNN和SVM训练样本

    # train_y, train_x = svmutil.svm_read_problem("/home/sunbite/train320240")
    # test_y, test_x = svmutil.svm_read_problem("/home/sunbite/test320240")
    # trainImages = train_x
    # trainLabels = train_y
    # testImages = test_x
    # testLabels = test_y
    trainImages_knn = trainImages.copy()
    trainLabels_knn = trainLabels.copy()
    trainImages_svm = trainImages.copy()
    trainLabels_svm = trainLabels.copy()

    testImages_knn = testImages.copy()
    testLabels_knn = testLabels.copy()
    testImages_svm = testImages.copy()
    testLabels_svm = testLabels.copy()

    # KNN和SVM用来测试的样本及测试的标签（不变）
    fixed_testImages = testImages.copy()
    fixed_testLabels = testLabels.copy()

    # 迭代次数

    # KNN和SVM保存准确率
    # KNN保存准确率
    accuracy_knn_list = []
    # SVM保存准确率
    accuracy_svm_list = []
    # Co_KNN_SVM保存准确率
    accuracy_Co_KNN_SVM_list = []

    # 协同训练
    for h in range(1, loop_num + 1):
        # KNN计算准确率
        testResults, accuracy_knn, Confidence = utilities.knn(trainImages_knn, trainLabels_knn, testImages_knn,
                                                             testLabels_knn, K)
        # testResults, accuary_knn, probility = utilities.knn_sklearn(trainImages_knn.copy(), trainLabels_knn.copy(), testImages_knn.copy(), testLabels_knn.copy(), K)

        # Confidence = []
        # for i in range(0, probility.shape[0]):
        #     temp = probility[i]
        #     Confidence.append(utilities.sub(temp))

        accuracy_knn_list.append(accuracy_knn * 100)

        trainImages_svmforlibsvm = utilities.getfeatureforlibsvm(trainImages_svm)
        # SVM计算准确率
        model = svmutil.svm_train(trainLabels_svm, trainImages_svmforlibsvm, '-s 0 -t 2 -c 15 -g 2 -b 1')

        # fixed_testImagesforlibsvm = utilities.getfeatureforlibsvm(fixed_testImages)
        # predict_label1, accuary1, prob_estimates1 = svmutil.svm_predict(fixed_testLabels, fixed_testImagesforlibsvm, model,
        #                                                                 '-b 1')
        testImages_svmforlibsvm = utilities.getfeatureforlibsvm(testImages_svm)
        predict_label, accuracy_svm, prob_estimates = svmutil.svm_predict(testLabels_svm, testImages_svmforlibsvm, model,
                                                                         '-b 1')
        accuracy_svm_list.append(accuracy_svm[0])
        accuracy_Co_KNN_SVM_list.append(accuracy_svm[0])

        testLength_svm = np.array(testImages_svm).shape[0]
        Confidence_SVM = []
        for i in range(0, testLength_svm):
            temp = prob_estimates[i]
            Confidence_SVM.append(utilities.sub_SVM(temp))

        if h == loop_num:
            svmutil.sa
            break
        # KNN和SVM半监督训练过程
        # ---------------------------------KNN测试样本预测和置信度计算过程 ----------------------------------
        # trainLength = np.array(trainImages_knn).shape[0]
        # testLength = np.array(testImages_knn).shape[0]
        # # 保存测试集的预测结果
        # testResults = np.linspace(0, 0, testLength)
        # # 保存预测置信度
        # Confidence = []
        #
        # Confidence, testResults = utilities.getConfidenceAndTestResults(trainImages_knn, trainLabels_knn,
        #                                                                 testImages_knn, K)

        # ---------------------------------SVM测试样本预测和置信度计算过程 ----------------------------------
        # 根据模型，预测样本
        # print("testLabels_svm的长度：")
        # print(len(testLabels_svm))
        # print("testImages_svm的长度：")
        # print(len(testImages_svm))
        # testImages_svmforlibsvm = utilities.getfeatureforlibsvm(testImages_svm)
        # predict_label, accuary, prob_estimates = svmutil.svm_predict(testLabels_svm, testImages_svmforlibsvm, model, '-b 1')

        # testLength_svm = np.array(testImages_svm).shape[0]
        # Confidence_SVM = []
        # for i in range(0, testLength_svm):
        #     temp = prob_estimates[i]
        #     Confidence_SVM.append(utilities.sub_SVM(temp))
        # KNN和SVM伪标签添加过程
        # ---------------------------------------KNN---------------------------------------------
        index_label_SVM = utilities.getConfidence_SVM(Confidence_SVM, predict_label, temp_num_SVM, testLength_svm)
        temp_test_train_SVM = []
        temp_test_label_SVM = []

        for i in index_label_SVM:
            temp_test_train_SVM.append(testImages_svm[i])
            temp_test_label_SVM.append(predict_label[i])

        trainImages_knn = np.concatenate((np.array(trainImages_knn), np.array(temp_test_train_SVM)))
        trainImages_knn = trainImages_knn.tolist()
        trainLabels_knn = np.concatenate((np.array(trainLabels_knn), np.array(temp_test_label_SVM)))
        trainLabels_knn = trainLabels_knn.tolist()
        # 获取新的测试样本
        temp_a = np.arange(0, testLength_svm)
        diff_temp_a = np.setdiff1d(temp_a, np.array(index_label_SVM))
        diff_testImages_svm = []
        diff_testLabels_svm = []
        for i in diff_temp_a:
            diff_testImages_svm.append(testImages_svm[i])
            diff_testLabels_svm.append(testLabels_svm[i])
        testImages_svm = diff_testImages_svm
        testLabels_svm = diff_testLabels_svm

        # ---------------------------------------SVM---------------------------------------------
        temp_label = utilities.getConfidence(Confidence, testResults, temp_num)

        temp_test_train = []
        temp_test_label = []

        for i in temp_label:
            temp_test_train.append(testImages_knn[i])
            temp_test_label.append(testResults[i])

        trainImages_svm = np.concatenate((np.array(trainImages_svm), np.array(temp_test_train)))
        trainImages_svm = trainImages_svm.tolist()
        trainLabels_svm = np.concatenate((np.array(trainLabels_svm), np.array(temp_test_label)))
        trainLabels_svm = trainLabels_svm.tolist()

        # 获取新的测试样本
        testLength = np.array(testImages_knn).shape[0]
        temp_b = np.arange(0, testLength)
        diff_temp_b = np.setdiff1d(temp_b, np.array(temp_label))
        diff_testImages_knn = []
        diff_testLabels_knn = []
        for i in diff_temp_b:
            diff_testImages_knn.append(testImages_knn[i])
            diff_testLabels_knn.append(testLabels_knn[i])
        testImages_knn = diff_testImages_knn
        testLabels_knn = diff_testLabels_knn
    print("Co_KNN_SVMde 准确率：")
    print(accuracy_Co_KNN_SVM_list)
    # plt.figure()
    # x = range(1, len(accuracy_svm_list) + 1)
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Accuracy OF CO_KNN_SVM")
    # plt.plot(x, accuracy_svm_list)
    # plt.show()
