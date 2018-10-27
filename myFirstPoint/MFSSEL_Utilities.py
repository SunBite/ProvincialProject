# -*- coding: utf-8 -*-
import numpy as np


def get_Y_X_Name_list_from_tuple(Y_X_tuple_list):
    """
    从标签和特征组成的元组list得到相应的标签list和特征list和名字list
    :param Y_X_tuple_list:标签和特征和名字组成的元组list
    :return:标签list，特征list，名字list
    """
    Y_list = []
    X_list = []
    Name_list = []
    if len(Y_X_tuple_list) is not 0:
        for i in range(len(Y_X_tuple_list)):
            Y_list.append(Y_X_tuple_list[i][0])
            X_list.append(Y_X_tuple_list[i][1])
            Name_list.append(Y_X_tuple_list[i][2])
        return Y_list, X_list, Name_list


def get_Y_X_Name_tuple_list(Y_list, X_list, Name_list):
    """
    返回标签和特征和名字组成的元组list
    :param Y_list: 标签list
    :param X_list: 特征list
    :param Name_list: 名字list
    :return:Y_X_tuple_list：标签和特征和名字组成的元组list
    """
    Y_X_Name_tuple_list = []
    if len(Y_list) == len(X_list) == len(Name_list):
        for i in range(len(Y_list)):
            Y_X_Name_tuple_list.append((Y_list[i], X_list[i]), Name_list[i])
        return Y_X_Name_tuple_list
    else:
        return Y_X_Name_tuple_list


def calc_ent(probilities):
    """
    计算信息熵
    :param probilities: 每个类别对应的概率
    :return:信息熵
    """
    ent = np.float64(0)
    for probility in probilities:
        ent -= probility * np.log2(probility)
    return ent


def get_confidence(probilities, para):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        sub_max_p = result[1]
        except_max_p_list = result[1:]
        sum_except_max_p = np.float64(0)
        for p in except_max_p_list:
            sum_except_max_p += p
        avg_except_max_p = sum_except_max_p / len(except_max_p_list)
        confidence = (1 - para) * (max_p - sub_max_p) + para * (max_p - avg_except_max_p)
    return confidence


def get_confidence_1(probilities):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        sub_max_p = result[1]
        confidence = max_p - sub_max_p
    return confidence


def get_confidence_2(probilities):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        sub_max_p = result[1]
        except_max_p_list = result[1:]
        sum_except_max_p = np.float64(0)
        for p in except_max_p_list:
            sum_except_max_p += p
        avg_except_max_p = sum_except_max_p / len(except_max_p_list)
        confidence = max_p - avg_except_max_p
    return confidence


def get_confidence_index(probilityList):
    """
    获取SVM置信度较高的索引
    :param probilityList:三个分类器对应的每个类别的概率
    :return:
    """
    # 每个类别的概率和预测结果
    hog_svc_probility = probilityList[0]
    _81_svc_probility = probilityList[1]
    _30_svc_probility = probilityList[2]
    # 每个分类器的置信度list
    hog_svc_confidence_list = []
    _81_svc_confidence_list = []
    _30_svc_confidence_list = []
    # 添加置信度list
    for i in hog_svc_probility:
        hog_svc_confidence_list.append(get_confidence(i, 0.9))
        # hog_svc_confidence_list.append(get_confidence_1(i))
        # hog_svc_confidence_list.append(get_confidence_2(i))
    for i in _81_svc_probility:
        _81_svc_confidence_list.append(get_confidence(i, 0.9))
        # _81_svc_confidence_list.append(get_confidence_1(i))
        # _81_svc_confidence_list.append(get_confidence_2(i))
    for i in _30_svc_probility:
        _30_svc_confidence_list.append(get_confidence(i, 0.9))
        # _30_svc_confidence_list.append(get_confidence_1(i))
        # _30_svc_confidence_list.append(get_confidence_2(i))
    hog_svc_confidence_list = np.array(hog_svc_confidence_list)
    _81_svc_confidence_list = np.array(_81_svc_confidence_list)
    _30_svc_confidence_list = np.array(_30_svc_confidence_list)
    # 置信度降序排列的序号
    hog_svc_ind_confidence_list = np.argsort(-hog_svc_confidence_list)
    _81_svc_ind_confidence_list = np.argsort(-_81_svc_confidence_list)
    _30_svc_ind_confidence_list = np.argsort(-_30_svc_confidence_list)
    # 置信度降序排列
    sorted_hog_svc_confidence_list = hog_svc_confidence_list[hog_svc_ind_confidence_list]
    sorted_81_svc_confidence_list = _81_svc_confidence_list[_81_svc_ind_confidence_list]
    sorted_30_svc_confidence_list = _30_svc_confidence_list[_30_svc_ind_confidence_list]

    return [(hog_svc_ind_confidence_list, sorted_hog_svc_confidence_list),
            (_81_svc_ind_confidence_list, sorted_81_svc_confidence_list),
            (_30_svc_ind_confidence_list, sorted_30_svc_confidence_list)]


def vote(predict_Y_list, real_unlabeled_Y):
    """
    针对三个分类起分类出来的结果进行投票
    :param predict_Y_list: 包含三个分类器的预测标签list
    :return: 投票之后的结果标签list
    """

    hog_svc_predict_Y, _81_svc_predict_Y, _30_svc_predict_Y = predict_Y_list
    hog_svc_predict_Y = np.array(hog_svc_predict_Y)
    _81_svc_predict_Y = np.array(_81_svc_predict_Y)
    _30_svc_predict_Y = np.array(_30_svc_predict_Y)

    hog_unlabeled_Y, _81_unlabeled_Y, _30_unlabeled_Y = real_unlabeled_Y

    voted_index_result = []
    voted_predict_Y_list = []
    for i in range(len(hog_svc_predict_Y)):
        if (hog_svc_predict_Y[i] == _81_svc_predict_Y[i] == _30_svc_predict_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (hog_svc_predict_Y[i] == _81_svc_predict_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (hog_svc_predict_Y[i] == _30_svc_predict_Y[i] == _30_unlabeled_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (_81_svc_predict_Y[i] == _30_svc_predict_Y[i] == _30_unlabeled_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(_81_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue

    return voted_index_result, voted_predict_Y_list


def get_voted_confidence(probility_list, voted_index_result, voted_predict_Y_list, whole_class, topk):
    """
    获得投票之后的经过置信度排序之后的排序index和置信度
    :param probility_list:
    :param voted_index_result:
    :return:
    """
    # 每个类别的概率和预测结果
    hog_svc_probility = probility_list[0]
    _81_svc_probility = probility_list[1]
    _30_svc_probility = probility_list[2]

    voted_hog_svc_probility = []
    voted_81_svc_probility = []
    voted_30_svc_probility = []

    for i in voted_index_result:
        voted_hog_svc_probility.append(hog_svc_probility[i])
        voted_81_svc_probility.append(_81_svc_probility[i])
        voted_30_svc_probility.append(_30_svc_probility[i])

    # 每个分类器的置信度list
    hog_svc_confidence_list = []
    _81_svc_confidence_list = []
    _30_svc_confidence_list = []

    # 添加置信度list
    for i in voted_hog_svc_probility:
        hog_svc_confidence_list.append(get_confidence(i, 0.9))#0.9
        #hog_svc_confidence_list.append(get_confidence_1(i))
        #hog_svc_confidence_list.append(get_confidence_2(i))
    for i in voted_81_svc_probility:
        _81_svc_confidence_list.append(get_confidence(i, 0.9))#0.9
        #_81_svc_confidence_list.append(get_confidence_1(i))
        #_81_svc_confidence_list.append(get_confidence_2(i))
    for i in voted_30_svc_probility:
        _30_svc_confidence_list.append(get_confidence(i, 0.9))#0.9
        #_30_svc_confidence_list.append(get_confidence_1(i))
        #_30_svc_confidence_list.append(get_confidence_2(i))

    hog_svc_confidence_list = np.array(hog_svc_confidence_list)
    _81_svc_confidence_list = np.array(_81_svc_confidence_list)
    _30_svc_confidence_list = np.array(_30_svc_confidence_list)

    e = hog_svc_confidence_list[np.argsort(-hog_svc_confidence_list)]
    f = _81_svc_confidence_list[np.argsort(-_81_svc_confidence_list)]
    g = _30_svc_confidence_list[np.argsort(-_30_svc_confidence_list)]

    voted_Y_list = []
    voted_index_list = []
    for i in range(whole_class):
        voted_Y_index = get_topk_Y_index(i + 1, topk, voted_predict_Y_list, voted_index_result, hog_svc_confidence_list,
                                         _81_svc_confidence_list, _30_svc_confidence_list)

        voted_Y_list.extend(voted_Y_index[0])
        voted_index_list.extend(voted_Y_index[1])
    a = []
    b = []
    c = []
    for i in voted_index_list:
        a.append(hog_svc_probility[i])
        b.append(_81_svc_probility[i])
        c.append(_30_svc_probility[i])
    return voted_Y_list, voted_index_list


def get_topk_Y_index(c, topk, voted_predict_Y_list, voted_index_result, hog_svc_confidence_list,
                     _81_svc_confidence_list, _30_svc_confidence_list):
    avg_confidence_list = []
    each_class_voted_index = []
    for i in range(len(voted_predict_Y_list)):
        if voted_predict_Y_list[i] == c:
            # print("hog_svc_confidence_list[i]:")
            # print(hog_svc_confidence_list[i])
            # print("_81_svc_confidence_list[i]:")
            # print(_81_svc_confidence_list[i])
            # print("_30_svc_confidence_list[i]:")
            # print(_30_svc_confidence_list[i])

            # avg_confidence = (hog_svc_confidence_list[i] + _81_svc_confidence_list[i] + _30_svc_confidence_list[i]) / 3
            avg_confidence = 0.6 * (hog_svc_confidence_list[i]) + 0.3 * (_81_svc_confidence_list[i]) + 0.1 * (
                _30_svc_confidence_list[i])
            avg_confidence_list.append(avg_confidence)
            each_class_voted_index.append(voted_index_result[i])

    if len(each_class_voted_index) < topk:
        topk = len(each_class_voted_index)

    avg_confidence_list = np.array(avg_confidence_list)
    each_class_voted_index = np.array(each_class_voted_index)
    ind_avg_confidence_list = np.argsort(-avg_confidence_list)
    sorted_avg_confidence_list = avg_confidence_list[ind_avg_confidence_list]
    sorted_each_voted_index_result = each_class_voted_index[ind_avg_confidence_list]
    voted_Y = []
    voted_index = []

    for i in range(topk):
        voted_Y.append(c)
        voted_index.append(sorted_each_voted_index_result[i])

    # avg_confidence_list = []
    # for i in range(len(hog_svc_confidence_list)):
    #     avg_confidence = 0.6 * (hog_svc_confidence_list[i]) + 0.3 * (_81_svc_confidence_list[i]) + 0.1 * (
    #         _30_svc_confidence_list[i])
    #     avg_confidence_list.append(avg_confidence)
    #
    # avg_confidence_list = np.array(avg_confidence_list)
    # voted_predict_Y_list = np.array(voted_predict_Y_list)
    # voted_index_result = np.array(voted_index_result)
    # ind_avg_confidence_list = np.argsort(-avg_confidence_list)
    # sorted_avg_confidence_list = avg_confidence_list[ind_avg_confidence_list]
    # sorted_voted_predict_Y_list = voted_predict_Y_list[ind_avg_confidence_list]
    # sorted_voted_index_result = voted_index_result[ind_avg_confidence_list]
    # voted_Y = []
    # voted_index = []
    #
    # for i in range(topk):
    #     voted_Y.append(sorted_voted_predict_Y_list[i])
    #     voted_index.append(sorted_voted_index_result[i])

    return voted_Y, voted_index
