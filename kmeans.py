#coding:utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class K_means(object):
    """docstring for K_means."""
    def __init__(self, n_clusters=2, max_iter=300):
        #super(K_means, self).__init__()
        #self.arg = arg
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers_ = None

    # ユークリッド距離を求める関数
    def euclidean_distance(self, x, p):
        return np.sum((x - p) ** 2)

    # クラスタリングをする関数
    def fit_predict(self, features):
        # 要素の中からセントロイド (重心) の初期値となる候補をクラスタ数だけ選び出す
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = features[initial_centroid_indexes]

        # ラベル付けした結果となる配列はゼロで初期化しておく
        pred = np.zeros(features.shape)

        # クラスタリングをアップデートする
        for _ in range(self.max_iter):

            # 各クラスタごとにセントロイド (重心) を計算する
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

            # 各特徴ベクトルから最短距離となる重心を基準に新しいラベルをつける
            new_pred = np.array([
                np.array([
                    self.euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])

            if np.all(new_pred == pred):
                # 更新前と内容を比較して、もし同じなら終了
                break

            pred = new_pred

        return pred

if __name__ == "__main__":

    K = 2  # クラスター数
    prefecture_data = pd.read_csv("./sangyohi.csv",header=None) # データ読み込み

    # Pandas のデータフレームから Numpy の行列 (Array) に変換
    p_array = np.array( [] )
    for i in range(len(prefecture_data.columns)):
        p_array = np.append(p_array,[prefecture_data[i].tolist()])

    p_array = p_array.T # 行列を転置

    print(p_array)
    # # クラスタリングする
    # cls = K_means(n_clusters = K)
    # pred = cls.fit_predict(p_array)
    #
    # # 各要素をラベルごとに色付けして表示する
    # for i in range(K):
    #     labels = p_array[pred == i]
    #     plt.scatter(labels[:, 0], labels[:, 1])
    #
    # centers = cls.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], s=100,
    #             facecolors='none', edgecolors='black')
    #
    # plt.show()
