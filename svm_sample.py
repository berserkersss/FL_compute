#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用于构造所需数据集，并计算P(x)
# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, metrics
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 用户数据集大小和标签种类
    num_img = [600, 600]
    num_label = [6, 8]

    X_train_user , y_train_user = [], []

    csv_path_train_data = 'csv/training_image.csv'
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    X_all_train = X_all_train.values

    csv_path_train_data = 'csv/training_label.csv'
    y_all_train = pd.read_csv(csv_path_train_data, header=None)
    y_all_train = y_all_train.values

    # 取数据并存数据
    csv_path_train_data = 'user_csv/svm/' + 'train' + '.csv'
    index = pd.read_csv(csv_path_train_data, header=None)
    index = index.values.T[0].astype(int)
    index_set = np.arange(1) + 1050
    index_set = np.hstack((np.arange(99),index_set))
    filename_idx = 'user_csv/svm/' + 'user' + str(0) + 'train_index_unbalance' + '.csv'
    np.savetxt(filename_idx, index[index_set], delimiter=',')
    X_train_user.append(X_all_train[index[index_set], :])
    y_train_user.append(y_all_train[index[index_set], :])

    csv_path_train_data = 'user_csv/svm/' + 'train' + '.csv'
    index = pd.read_csv(csv_path_train_data, header=None)
    index = index.values.T[0].astype(int)
    index_set = np.arange(99) + 600
    index_set = np.hstack(((np.arange(1) + 450), index_set))
    filename_idx = 'user_csv/svm/' + 'user' + str(1) + 'train_index_unbalance' + '.csv'
    np.savetxt(filename_idx, index[index_set], delimiter=',')
    X_train_user.append(X_all_train[index[index_set], :])
    y_train_user.append(y_all_train[index[index_set], :])

    csv_path_train_data = 'user_csv/svm/' + 'train' + '.csv'
    index = pd.read_csv(csv_path_train_data, header=None)
    index = index.values.T[0].astype(int)
    index_set = np.arange(100) + 500
    index_set = np.hstack(((np.arange(100) + 1100), index_set))
    filename_idx = 'user_csv/svm/' + 'test' + '.csv'
    np.savetxt(filename_idx, index[index_set], delimiter=',')

    # 抽取总数据
    csv_path_train_data = 'user_csv/svm/' + 'train' + '.csv'
    train_index = pd.read_csv(csv_path_train_data, header=None)
    train_index = train_index.values
    train_index = train_index.T[0].astype(int)
    X_all_train = X_all_train[train_index, :]
    y_all_train = y_all_train[train_index]

    Px_all = np.zeros((1, X_all_train.shape[1]))
    Px_user = np.zeros((2, X_all_train.shape[1]))

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
