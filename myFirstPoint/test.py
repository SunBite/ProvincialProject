# -*- coding: utf-8 -*-
from myFirstPoint.DataPreparation import DataPreparation
import myFirstPoint.MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
import numpy as np
import math
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog, ORB
import cv2 as cv

orbFeatureDir = "/home/sunbite/features_new/orbvideoFeature/"
hogFeatureDir = "/home/sunbite/features_new/hogvideoFeature/"
_81FeatureDir = "/home/sunbite/features/_81videoFeature/"
_30FeatureDir = "/home/sunbite/features_new/_30videoFeature/"
_5FeatureDir = "/home/sunbite/features_new/_5videoFeature/"
_6FeatureDir = "/home/sunbite/features_new/_6videoFeature/"
# 初始化dataPreparation对象
dataPreparation = DataPreparation()
# 获取有标签训练集，无标签训练集，测试集的标签和名字tuplelist
label_name_labeled_train_tuple_list, label_name_unlabeled_train_tuple_list, label_name_test_tuple_list = dataPreparation.getLabelAndNameTupleList(
    "/home/sunbite/features/_122videoFeature/")

# # 获取hog维有标签训练集
# hog_labeled_train_tuple_list = dataPreparation.loadData(hogFeatureDir, label_name_labeled_train_tuple_list)
# hog_labeled_train_Y, hog_labeled_train_X, hog_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     hog_labeled_train_tuple_list)
# # 获取hog维无标签训练集
# hog_unlabeled_tuple_list = dataPreparation.loadData(hogFeatureDir, label_name_unlabeled_train_tuple_list)
# hog_unlabeled_Y, hog_unlabeled_X, hog_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     hog_unlabeled_tuple_list)
# # 获取hog维测试集
# hog_test_tuple_list = dataPreparation.loadData(hogFeatureDir, label_name_test_tuple_list)
# hog_test_Y, hog_test_X, hog_test_Name = Utilities.get_Y_X_Name_list_from_tuple(hog_test_tuple_list)
# hogsvc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # gamma=12 c=4
# hogsvc_1.fit(hog_unlabeled_X, hog_unlabeled_Y)
# # 获得准确率
# hogaccuracy = hogsvc_1.score(hog_test_X, hog_test_Y)


# # 获取81维有标签训练集
# _81_labeled_train_tuple_list = dataPreparation.loadData(_81FeatureDir, label_name_labeled_train_tuple_list)
# _81_labeled_train_Y, _81_labeled_train_X, _81_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _81_labeled_train_tuple_list)
# # 获取81维无标签训练集
# _81_unlabeled_tuple_list = dataPreparation.loadData(_81FeatureDir, label_name_unlabeled_train_tuple_list)
# _81_unlabeled_Y, _81_unlabeled_X, _81_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _81_unlabeled_tuple_list)
# # 获取81维测试集
# _81_test_tuple_list = dataPreparation.loadData(_81FeatureDir, label_name_test_tuple_list)
# _81_test_Y, _81_test_X, _81_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_81_test_tuple_list)
#
# # 81维svm训练
# _81svc_1 = SVC(C=2, kernel='rbf', gamma=6, probability=True)  # gamma=7
# _81svc_1.fit(_81_unlabeled_X, _81_unlabeled_Y)
# # 获得准确率
# _81accuracy = _81svc_1.score(_81_test_X, _81_test_Y)

# 获取30维有标签训练集
_30_labeled_train_tuple_list = dataPreparation.loadData(_30FeatureDir, label_name_labeled_train_tuple_list)
_30_labeled_train_Y, _30_labeled_train_X, _30_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
    _30_labeled_train_tuple_list)
# 获取30维无标签训练集
_30_unlabeled_tuple_list = dataPreparation.loadData(_30FeatureDir, label_name_unlabeled_train_tuple_list)
_30_unlabeled_Y, _30_unlabeled_X, _30_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
    _30_unlabeled_tuple_list)
# 获取30维测试集
_30_test_tuple_list = dataPreparation.loadData(_30FeatureDir, label_name_test_tuple_list)
_30_test_Y, _30_test_X, _30_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_30_test_tuple_list)
# _30svc_1 = SVC(C=32, kernel='rbf', gamma=12, probability=True)  # gamma=7
# _30svc_1.fit(_30_unlabeled_X, _30_unlabeled_Y)
# # 获得准确率
# _30accuracy = _30svc_1.score(_30_test_X, _30_test_Y)

# # 获取_5维有标签训练集
# _5_labeled_train_tuple_list = dataPreparation.loadData(_5FeatureDir, label_name_labeled_train_tuple_list)
# _5_labeled_train_Y, _5_labeled_train_X, _5_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _5_labeled_train_tuple_list)
# # 获取_5维无标签训练集
# _5_unlabeled_tuple_list = dataPreparation.loadData(_5FeatureDir, label_name_unlabeled_train_tuple_list)
# _5_unlabeled_Y, _5_unlabeled_X, _5_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _5_unlabeled_tuple_list)
# # 获取_5维测试集
# _5_test_tuple_list = dataPreparation.loadData(_5FeatureDir, label_name_test_tuple_list)
# _5_test_Y, _5_test_X, _5_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_5_test_tuple_list)
#

# # 获取_6维有标签训练集
# _6_labeled_train_tuple_list = dataPreparation.loadData(_6FeatureDir, label_name_labeled_train_tuple_list)
# _6_labeled_train_Y, _6_labeled_train_X, _6_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _6_labeled_train_tuple_list)
# # 获取_6维无标签训练集
# _6_unlabeled_tuple_list = dataPreparation.loadData(_6FeatureDir, label_name_unlabeled_train_tuple_list)
# _6_unlabeled_Y, _6_unlabeled_X, _6_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     _6_unlabeled_tuple_list)
# # 获取_6维测试集
# _6_test_tuple_list = dataPreparation.loadData(_6FeatureDir, label_name_test_tuple_list)
# _6_test_Y, _6_test_X, _6_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_6_test_tuple_list)
#
# accuracy_svm_list = []
# C = [1, 2, 4, 8, 16, 32, 64, 128]
# gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# for i in C:
#     for j in gamma:
#         # 30维svm训练
#         _6svc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
#         _6svc_1.fit(_6_unlabeled_X, _6_unlabeled_Y)
#         # 获得准确率
#         _6accuracy = _6svc_1.score(_6_test_X, _6_test_Y)
#
#         accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(_6accuracy))
# print(accuracy_svm_list)
#
# accuracy_svm_list = []
# C = [1, 2, 4, 8, 16, 32, 64, 128]
# gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# for i in C:
#     for j in gamma:
#         # 30维svm训练
#         _5svc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
#         _5svc_1.fit(_5_unlabeled_X, _5_unlabeled_Y)
#         # 获得准确率
#         _5accuracy = _5svc_1.score(_5_test_X, _5_test_Y)
#
#         accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(_5accuracy))
# print(accuracy_svm_list)
_30_labeled_train_X.extend(_30_unlabeled_X)
_30_labeled_train_Y.extend(_30_unlabeled_Y)
accuracy_svm_list = []
C = [1, 2, 4, 8, 16, 32, 64, 128]
gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
for i in C:
    for j in gamma:
        # 30维svm训练
        _30svc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
        _30svc_1.fit(_30_unlabeled_X, _30_unlabeled_Y)
        # 获得准确率
        _30accuracy = _30svc_1.score(_30_test_X, _30_test_Y)
        print("c:" + str(i) + "   gamma:" + str(j) + "   accuracy:   " + str(_30accuracy))
        accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(_30accuracy))
print(accuracy_svm_list)
# _81_labeled_train_X.extend(_81_unlabeled_X)
# _81_labeled_train_Y.extend(_81_unlabeled_Y)
# accuracy_svm_list = []
# C = [1, 2, 4, 8, 16, 32, 64, 128]
# gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# for i in C:
#     for j in gamma:
#         # 81维svm训练
#         _81svc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
#         _81svc_1.fit(_81_labeled_train_X, _81_labeled_train_Y)
#         # 获得准确率
#         _81accuracy = _81svc_1.score(_81_test_X, _81_test_Y)
#         print("c:"+str(i)+"   gamma:"+str(j)+"   accuracy:   "+str(_81accuracy))
#         accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(_81accuracy))
# print(accuracy_svm_list)
# hog_labeled_train_X.extend(hog_unlabeled_X)
# hog_labeled_train_Y.extend(hog_unlabeled_Y)
# accuracy_svm_list = []
# C = [1, 2, 4, 8, 16, 32, 64, 128]
# gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# for i in C:
#     for j in gamma:
#
#         # hog维svm训练
#         hogsvc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=12 c=4
#         hogsvc_1.fit(hog_labeled_train_X, hog_labeled_train_Y)
#         # 获得准确率
#         hogaccuracy = hogsvc_1.score(hog_test_X, hog_test_Y)
#         accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(hogaccuracy))
# print(accuracy_svm_list)
# image = cv.imread('/home/sunbite/Co_KNN_SVM_TMP/keyframe/basketball_v_shooting_01_01_keyFrame_0.jpg')
# #image = cv.resize("/home/sunbite/Co_KNN_SVM_TMP/keyframe", (32, 32))
# image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# a = ORB(n_keypoints=200)
# a.detect_and_extract(image_gray)
# b= a.descriptors
# c= a.keypoints
# # 初始化dataPreparation对象
# dataPreparation = DataPreparation()
# # 获取有标签训练集，无标签训练集，测试集的标签和名字tuplelist
# label_name_labeled_train_tuple_list, label_name_unlabeled_train_tuple_list, label_name_test_tuple_list = dataPreparation.getLabelAndNameTupleList(
#     "/home/sunbite/features/_122videoFeature/")
#
# # 获取orb维有标签训练集
# orb_labeled_train_tuple_list = dataPreparation.loadData(orbFeatureDir, label_name_labeled_train_tuple_list)
# orb_labeled_train_Y, orb_labeled_train_X, orb_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     orb_labeled_train_tuple_list)
# # 获取orb维无标签训练集
# orb_unlabeled_tuple_list = dataPreparation.loadData(orbFeatureDir, label_name_unlabeled_train_tuple_list)
# orb_unlabeled_Y, orb_unlabeled_X, orb_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
#     orb_unlabeled_tuple_list)
# # 获取orb维测试集
# orb_test_tuple_list = dataPreparation.loadData(orbFeatureDir, label_name_test_tuple_list)
# orb_test_Y, orb_test_X, orb_test_Name = Utilities.get_Y_X_Name_list_from_tuple(orb_test_tuple_list)
# accuracy_svm_list = []
# C = [1, 2, 4, 8, 16, 32, 64, 128]
# gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# for i in C:
#     for j in gamma:
#         # 30维svm训练
#         orbsvc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
#         orbsvc_1.fit(orb_unlabeled_X, orb_unlabeled_Y)
#         # 获得准确率
#         orbaccuracy = orbsvc_1.score(orb_test_X, orb_test_Y)
#
#         accuracy_svm_list.append("c:"+str(i)+"   gamma:"+str(j)+"   accuracy"+str(orbaccuracy))
# print(accuracy_svm_list)

# orbsvc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # gamma=12 c=4
# orbsvc_1.fit(orb_unlabeled_X, orb_unlabeled_Y)
# # 获得准确率
# orbaccuracy = orbsvc_1.score(orb_test_X, orb_test_Y)
1 == 1
