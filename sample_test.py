#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用于构造所需测试集，并计算P(x)
# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, metrics
import math
import matplotlib.pyplot as plt


def mnist_noniid(train_labels, test_labels, num_label, train_num_img, test_num_img):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    sample_label = set(np.random.choice(range(len(set(train_labels.flatten()))), num_label, replace=False))

    train = 6000
    dict_users = np.array([], dtype='int64')
    idxs = np.arange(len(train_labels.flatten()))

    # sort labels
    idxs_labels = np.vstack((idxs, train_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for label in sample_label:
        sample_img = (np.random.choice(range(train), int(train_num_img / num_label), replace=False) + train * label)
        dict_users = np.concatenate((dict_users, idxs[sample_img]), axis=0)

    # test sample
    test = 1000
    idxs = np.arange(len(test_labels.flatten()))

    # sort labels
    idxs_labels = np.vstack((idxs, test_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    dict_users_test = np.array([], dtype='int64')
    for label in sample_label:
        sample_img = (
            np.random.choice(range(test), int(test_num_img / num_label), replace=False) + test * label)
        dict_users_test = np.concatenate((dict_users_test, idxs[sample_img]), axis=0)

    return dict_users, dict_users_test


if __name__ == '__main__':
    csv_path_train_data = 'csv/training_image.csv'
    csv_path_train_label = 'csv/training_label.csv'
    csv_path_test_data = 'csv/test_image.csv'
    csv_path_test_label = 'csv/test_label.csv'

    # 设置用户数据集大小和标签种类
    train_num_img = [1000, 600, 600, 400, 400]
    test_num_img = [400, 240, 240, 160, 160]
    num_label = [2, 3, 1, 2, 8]
    option = ['balance', 'unbalance']
    myoption = int(input("Enter the value for option(balance:0 unbalance:1): "))
    if myoption == 0:
        label = option[0]
    else:
        label = option[1]
    # 导入MNIST数据集，数据集由 60,000 个训练样本和 10,000 个测试样本组成，每个样本
    # 为一个28*28的图片，读入时我们将这个图片转换为1*784的向量
    # header=None 表示文件一开始就是数据
    X_all_test = pd.read_csv(csv_path_test_data, header=None)
    y_all_test = pd.read_csv(csv_path_test_label, header=None)
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    y_all_train = pd.read_csv(csv_path_train_label, header=None)

    # In[3]:
    X_all_test = X_all_test.values
    y_all_test = y_all_test.values
    # In[3]:
    X_all_train = X_all_train.values
    y_all_train = y_all_train.values

    test_inset = np.array([], dtype='int64')
    for k in range(len(num_label)):
        while True:
            index_set, index_set_test = mnist_noniid(y_all_train.T, y_all_test.T, num_label[k], train_num_img[k], test_num_img[k])
            X_train = X_all_train[index_set, :]
            y_train = y_all_train[index_set, :]
            y_test = y_all_test[index_set_test, :]
            if len(set(y_train.flatten())) == num_label[k]:  # 保证标签的个数，否则重新抽取
                filename_idx = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + label + '.csv'
                np.savetxt(filename_idx, index_set, delimiter=',')
                test_inset = np.hstack((test_inset, index_set_test))
                break
    filename_idx = 'user_csv/index/' + 'user_' + 'test_index_' + label + '.csv'
    np.savetxt(filename_idx, test_inset, delimiter=',')

    # filename = 'user_csv/' + 'user' + str(k) + 'train_label' + label + '.csv'
    # filename1 ='user_csv/' + 'user' + str(k) + 'train_img' + label + '.csv'
    # np.savetxt(filename1, X_train, delimiter=',')
    # np.savetxt(filename, y_train, delimiter=',')
    csv_path_train_data = 'csv/training_image.csv'
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    X_all_train = X_all_train.values

    X_train_user, y_train_user = [], []
    X_temp = np.array([], dtype='int64')
    for k in range(len(train_num_img)):
        csv_path_train_data = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + label + '.csv'
        train_index = pd.read_csv(csv_path_train_data, header=None)

        # 修剪数据集使得只有图片和标签,把序号剔除
        train_index = train_index.values
        train_index = train_index.T[0].astype(int)
        X_train_user.append(X_all_train[train_index, :])
        y_train_user.append(y_all_train[train_index])
        if k == 0:
            X_temp = X_all_train[train_index, :]
        else:
            X_temp = np.vstack((X_temp, X_all_train[train_index, :]))

    X_all_train = X_temp
    Px_all = np.zeros((1, X_all_train.shape[1]))
    Px_user = np.zeros((5, X_all_train.shape[1]))

    Px_all_gray = np.arange(784)
    Px_user_gray = {i: np.arange(784) for i in range(len(X_train_user))}
    for gray in range(256):
        for i in range(X_all_train.shape[1]):
            Px_all[0, i] = np.sum(X_all_train[:, i] == gray)
            Px_all[0, i] = Px_all[0, i] / 60000.0
            for j in range(len(X_train_user)):
                temp = X_train_user[j]
                Px_user[j, i] = np.sum(temp[:, i] == gray)
                Px_user[j, i] = 1.0 * Px_user[j, i] / len(temp[:, 0])
        for j in range(len(X_train_user)):
            Px_user_gray[j] = np.vstack((Px_user_gray[j], Px_user[j, :]))
        Px_all_gray = np.vstack((Px_all_gray, Px_all))

    Px_all_gray = Px_all_gray[1:, :]
    D_distribution = []
    D = []
    for idx in range(len(X_train_user)):
        Px = Px_user_gray[idx][1:, :]
        diff = Px_all_gray - Px
        D_distribution.append(np.linalg.norm(diff, ord=1))
        temp = np.linalg.norm(diff, ord=2, axis=0)
        D.append(np.linalg.norm(temp, ord=1))

    print(D_distribution)
    print(D)

    # compute Ld
    mu = 3.5169
    sigma = 0.2405
    Lm = []
    Ld = []
    for i in range(len(train_num_img)):
        Lm.append(1/(4*sigma**2/train_num_img[i] + 2*(mu*D[i])**2))
    for i in range(len(train_num_img)):
        Ld.append(Lm[i]/(sum(Lm)))

    print(Ld)



