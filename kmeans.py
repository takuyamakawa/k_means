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
    def euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)

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
            # 各要素から最短距離のセントロイドを基準にラベルを更新する
            new_pred = np.array([
                np.array([
                    self.euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])

            if np.all(new_pred == pred):
                # 更新前と内容が同じなら終了
                break

            pred = new_pred

            # 各クラスタごとにセントロイド (重心) を再計算する
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

        return pred

if __name__ == "__main__":

    K = 2  # クラスター数
    prefecture_data = pd.read_csv("./sangyohi.csv",header=None) # データ読み込み

    # Pandas のデータフレームから Numpy の行列 (Array) に変換
    p_array = np.array([prefecture_data[0].tolist(),
                        prefecture_data[1].tolist(),
                        prefecture_data[2].tolist(),
                        prefecture_data[3].tolist(),
                        prefecture_data[4].tolist(),
                        prefecture_data[5].tolist(),
                        prefecture_data[6].tolist(),
                        prefecture_data[7].tolist(),
                        prefecture_data[8].tolist(),
                        prefecture_data[9].tolist(),
                        prefecture_data[10].tolist(),
                        prefecture_data[11].tolist(),
                        prefecture_data[12].tolist(),
                        prefecture_data[13].tolist(),
                        prefecture_data[14].tolist(),
                        prefecture_data[15].tolist(),
                        prefecture_data[16].tolist(),
                        prefecture_data[17].tolist(),
                        prefecture_data[18].tolist(),
                        prefecture_data[19].tolist(),
                        prefecture_data[20].tolist(),
                        prefecture_data[21].tolist(),
                        prefecture_data[22].tolist(),
                        prefecture_data[23].tolist(),
                        prefecture_data[24].tolist(),
                        prefecture_data[25].tolist(),
                        prefecture_data[26].tolist(),
                        prefecture_data[27].tolist(),
                        prefecture_data[28].tolist(),
                       ], np.float64)

    p_array = p_array.T # 行列を転置
    # クラスタリングする
    cls = K_means(n_clusters = K)
    pred = cls.fit_predict(p_array)

    prefecture_data['29']=pred
    print(pred)
    print(prefecture_data['29'].value_counts())

    clusterinfo = pd.DataFrame()
    for i in range(K):
        clusterinfo['cluster' + str(i)] = prefecture_data[prefecture_data['29'] == i].mean()
    clusterinfo = clusterinfo.drop('29')

    my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 2 Clusters")
    my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)
    plt.show()
