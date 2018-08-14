# -*- coding: utf-8 -*-
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
