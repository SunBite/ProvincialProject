# -*- coding: utf-8 -*-
from myFirstPoint.DataPreparation import DataPreparation
import myFirstPoint.MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
from sklearn.externals import joblib
import os


def MFSSEL(featureNameDirPath, savePath=None, trainAndTestFlag="train"):
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
    topk = 5
    # 迭代次数
    loop_num = 100
    # 训练
    if trainAndTestFlag == "train":
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

        # 获取81维有标签训练集
        _81_labeled_train_tuple_list = dataPreparation.loadData(featureDir, train_81FeatureDir,
                                                                label_name_labeled_train_tuple_list)
        _81_labeled_train_Y, _81_labeled_train_X, _81_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _81_labeled_train_tuple_list)
        # 获取81维无标签训练集
        _81_unlabeled_tuple_list = dataPreparation.loadData(featureDir, train_81FeatureDir,
                                                            label_name_unlabeled_train_tuple_list)
        _81_unlabeled_Y, _81_unlabeled_X, _81_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _81_unlabeled_tuple_list)
        # 获取81维测试集
        _81_test_tuple_list = dataPreparation.loadData(featureDir, test_81FeatureDir, label_name_test_tuple_list)
        _81_test_Y, _81_test_X, _81_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_81_test_tuple_list)

        # 获取30维有标签训练集
        _30_labeled_train_tuple_list = dataPreparation.loadData(featureDir, train_30FeatureDir,
                                                                label_name_labeled_train_tuple_list)
        _30_labeled_train_Y, _30_labeled_train_X, _30_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _30_labeled_train_tuple_list)
        # 获取30维无标签训练集
        _30_unlabeled_tuple_list = dataPreparation.loadData(featureDir, train_30FeatureDir,
                                                            label_name_unlabeled_train_tuple_list)
        _30_unlabeled_Y, _30_unlabeled_X, _30_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _30_unlabeled_tuple_list)
        # 获取30维测试集
        _30_test_tuple_list = dataPreparation.loadData(featureDir, test_30FeatureDir, label_name_test_tuple_list)
        _30_test_Y, _30_test_X, _30_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_30_test_tuple_list)
        # hog_labeled_train_X.extend(hog_unlabeled_X)
        # hog_labeled_train_Y.extend(hog_unlabeled_Y)
        # _81_labeled_train_X.extend(_81_unlabeled_X)
        # _81_labeled_train_Y.extend(_81_unlabeled_Y)
        # _30_labeled_train_X.extend(_30_unlabeled_X)
        # _30_labeled_train_Y.extend(_30_unlabeled_Y)

        for h in range(1, loop_num + 1):
            # hog维svm训练
            hog_svc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # c:4 gamma=2
            hog_svc_1.fit(hog_labeled_train_X, hog_labeled_train_Y)

            # 81维svm训练
            _81_svc_1 = SVC(C=4, kernel='rbf', gamma=14, probability=True)  # c:2 gamma=6
            _81_svc_1.fit(_81_labeled_train_X, _81_labeled_train_Y)

            # 30维svm训练
            _30_svc_1 = SVC(C=32, kernel='rbf', gamma=8, probability=True)  # c:32 gamma=12
            _30_svc_1.fit(_30_labeled_train_X, _30_labeled_train_Y)

            # 获得准确率
            hog_accuracy = hog_svc_1.score(hog_test_X, hog_test_Y)
            # 获得准确率
            _81_accuracy = _81_svc_1.score(_81_test_X, _81_test_Y)
            # 获得准确率
            _30_accuracy = _30_svc_1.score(_30_test_X, _30_test_Y)
            # hog_accuracy，_81_accuracy，_30_accuracy中最大的accuracy
            accuracy_max_from_single = max(hog_accuracy, _81_accuracy, _30_accuracy)
            accuracy_list.append(accuracy_max_from_single * 100)
            # hog_accuracy_list.append(hog_accuracy * 100)
            # _81_accuracy_list.append(_81_accuracy * 100)
            # _30_accuracy_list.append(_30_accuracy * 100)
            print("accuracy_list:")
            print(accuracy_list)

            if accuracy_max_from_single > accuracy_max:
                accuracy_max = accuracy_max_from_single
                model_max_hog = hog_svc_1
                model_max_81 = _81_svc_1
                model_max_30 = _30_svc_1

            if len(hog_unlabeled_X) == 0 or len(_81_unlabeled_X) == 0 or len(_30_unlabeled_X) == 0:
                break
            if h == loop_num:
                break

            # hog特征下的无标签数据集的所对应的各个类别的概率
            hog_svc_1_probility = hog_svc_1.predict_proba(hog_unlabeled_X)
            # hog特征下无标签数据的预测标签
            hog_svc_1_predict_Y = hog_svc_1.predict(hog_unlabeled_X)

            # 81维特征下的无标签数据集的所对应的各个类别的概率
            _81_svc_1_probility = _81_svc_1.predict_proba(_81_unlabeled_X)
            # 81维特征下无标签数据的预测标签
            _81_svc_1_predict_Y = _81_svc_1.predict(_81_unlabeled_X)

            # 30维特征下的无标签数据集的所对应的各个类别的概率
            _30_svc_1_probility = _30_svc_1.predict_proba(_30_unlabeled_X)
            # 30维特征下无标签数据的预测标签

            _30_svc_1_predict_Y = _30_svc_1.predict(_30_unlabeled_X)
            probility_list_1 = [hog_svc_1_probility, _81_svc_1_probility, _30_svc_1_probility]
            unlabeled_Y_list_1 = [hog_unlabeled_Y, _81_unlabeled_Y, _30_unlabeled_Y]
            predict_Y_list_1 = [hog_svc_1_predict_Y, _81_svc_1_predict_Y, _30_svc_1_predict_Y]

            # ---------------------------------------------想法3---------------------------------------------------
            # voted_index_result, voted_predict_Y_list = Utilities.vote(predict_Y_list_1, unlabeled_Y_list_1)
            #
            # hog_real_Y = []
            # _81_real_Y = []
            # _30_real_Y = []
            # for i in voted_index_result:
            #     hog_real_Y.append(hog_unlabeled_Y[i])
            #     _81_real_Y.append(_81_unlabeled_Y[i])
            #     _30_real_Y.append(_30_unlabeled_Y[i])
            # # print(hog_real_Y)
            # # print(voted_predict_Y_list)
            # for i in range(len(voted_index_result)):
            #     hog_labeled_train_X.append(hog_unlabeled_X[voted_index_result[i]])
            #     hog_labeled_train_Y.append(voted_predict_Y_list[i])
            #     _81_labeled_train_X.append(_81_unlabeled_X[voted_index_result[i]])
            #     _81_labeled_train_Y.append(voted_predict_Y_list[i])
            #     _30_labeled_train_X.append(_30_unlabeled_X[voted_index_result[i]])
            #     _30_labeled_train_Y.append(voted_predict_Y_list[i])
            #
            # hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in voted_index_result]
            # hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in voted_index_result]
            # _81_unlabeled_X = [i for j, i in enumerate(_81_unlabeled_X) if j not in voted_index_result]
            # _81_unlabeled_Y = [i for j, i in enumerate(_81_unlabeled_Y) if j not in voted_index_result]
            # _30_unlabeled_X = [i for j, i in enumerate(_30_unlabeled_X) if j not in voted_index_result]
            # _30_unlabeled_Y = [i for j, i in enumerate(_30_unlabeled_Y) if j not in voted_index_result]
            # print(len(hog_labeled_train_Y))
            # print(len(hog_unlabeled_Y))
            # print(hog_unlabeled_Y)
            # ----------------------------------------------想法2----------------------------------------------------
            voted_index_predict_Y_list = Utilities.vote(predict_Y_list_1, unlabeled_Y_list_1, whole_class, topk)

            voted_Y_list, voted_index_list = Utilities.get_voted_confidence(probility_list_1,
                                                                            voted_index_predict_Y_list[0],
                                                                            voted_index_predict_Y_list[1], whole_class,
                                                                            topk)

            hog_real_Y = []
            _81_real_Y = []
            _30_real_Y = []
            for i in voted_index_list:
                hog_real_Y.append(hog_unlabeled_Y[i])
                _81_real_Y.append(_81_unlabeled_Y[i])
                _30_real_Y.append(_30_unlabeled_Y[i])
            print(hog_real_Y)
            print(voted_Y_list)
            for i in range(len(voted_index_list)):
                hog_labeled_train_X.append(hog_unlabeled_X[voted_index_list[i]])
                hog_labeled_train_Y.append(voted_Y_list[i])
                _81_labeled_train_X.append(_81_unlabeled_X[voted_index_list[i]])
                _81_labeled_train_Y.append(voted_Y_list[i])
                _30_labeled_train_X.append(_30_unlabeled_X[voted_index_list[i]])
                _30_labeled_train_Y.append(voted_Y_list[i])
            print(len(hog_labeled_train_Y))

            hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in voted_index_list]
            hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in voted_index_list]
            _81_unlabeled_X = [i for j, i in enumerate(_81_unlabeled_X) if j not in voted_index_list]
            _81_unlabeled_Y = [i for j, i in enumerate(_81_unlabeled_Y) if j not in voted_index_list]
            _30_unlabeled_X = [i for j, i in enumerate(_30_unlabeled_X) if j not in voted_index_list]
            _30_unlabeled_Y = [i for j, i in enumerate(_30_unlabeled_Y) if j not in voted_index_list]

            # ---------------------------------------------想法1------------------------------
            # index = random.sample(range(len(hog_unlabeled_Y)), 55)
            #
            # for i in index:
            #     hog_labeled_train_Y.append(hog_unlabeled_Y[i])
            #     hog_labeled_train_X.append(hog_unlabeled_X[i])
            #     _81_labeled_train_Y.append(_81_unlabeled_Y[i])
            #     _81_labeled_train_X.append(_81_unlabeled_X[i])
            #     _30_labeled_train_Y.append(_30_unlabeled_Y[i])
            #     _30_labeled_train_X.append(_30_unlabeled_X[i])
            # hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in index]
            # hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in index]
            # _81_unlabeled_X = [i for j, i in enumerate(_81_unlabeled_X) if j not in index]
            # _81_unlabeled_Y = [i for j, i in enumerate(_81_unlabeled_Y) if j not in index]
            # _30_unlabeled_X = [i for j, i in enumerate(_30_unlabeled_X) if j not in index]
            # _30_unlabeled_Y = [i for j, i in enumerate(_30_unlabeled_Y) if j not in index]
        print(accuracy_max * 100)
        if model_max_hog is not None:
            print("正在保存hog.model...")
            joblib.dump(model_max_hog, savePath + "hog.model")
            print("保存hog.model完毕。")
        if model_max_81 is not None:
            print("正在保存81.model...")
            joblib.dump(model_max_81, savePath + "81.model")
            print("保存81.model完毕。")
        if model_max_30 is not None:
            print("正在保存30.model...")
            joblib.dump(model_max_30, savePath + "30.model")
            print("保存30.model完毕。")
    # 加载训练好的model，进行测试
    elif trainAndTestFlag == "test":
        if os.path.exists(savePath):
            # 加载model文件
            model_max_hog = joblib.load(savePath + "hog.model")
            model_max_81 = joblib.load(savePath + "81.model")
            model_max_30 = joblib.load(savePath + "30.model")


if __name__ == '__main__':
    MFSSEL("/home/sunbite/MFSSEL/features_new_1/", "/home/sunbite/MFSSEL/model/")
