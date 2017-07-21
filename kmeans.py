#coding:utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

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
        return np.linalg.norm(p0 - p1)

    # 事例ベクトルと代表ベクトルの総和を求める関数
    def all_vector_distance(self, prefecture_data, clusters, centroid):
        dists = 0.0
        for centroid_num in range(centroid.shape[0]):
            for clusters_num in range(clusters.shape[0]):
                if clusters[clusters_num] == centroid_num:
                    dists += self.euclidean_distance(centroid[centroid_num], prefecture_data[clusters_num])
        return dists

    # クラスタリングをする関数
    def fit_predict(self, prefecture_data):
        # 要素の中からセントロイド (重心) の初期値となる候補をクラスタ数だけ選び出す
        feature_indexes = np.arange(len(prefecture_data))
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = prefecture_data[initial_centroid_indexes]

        # ラベル付けした結果となる配列はゼロで初期化しておく
        cluster = np.zeros(prefecture_data.shape)

        init_distance = 10000
        result = None

        for _ in range(1000):

            # クラスタリングをアップデートする
            for _ in range(self.max_iter):
                # 各要素から最短距離のセントロイドを基準にラベルを更新する
                new_cluster = np.array([
                    np.array([
                        self.euclidean_distance(p, centroid)
                        for centroid in self.cluster_centers_
                    ]).argmin()
                    for p in prefecture_data
                ])

                if np.all(new_cluster == cluster):
                    # 更新前と内容が同じなら終了
                    break

                cluster = new_cluster

                # 事例ベクトルと代表ベクトルの総和を出して、最小となるものを結果とする
                current_distance = self.all_vector_distance(prefecture_data, cluster, self.cluster_centers_)
                if current_distance < init_distance:
                    init_distance = current_distance
                    result = cluster

        return result

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
    result = cls.fit_predict(p_array)

    prefecture_data['29']=result
    print(result)
    print(prefecture_data['29'].value_counts())

    clusterinfo = pd.DataFrame()
    for i in range(K):
        clusterinfo['cluster' + str(i)] = prefecture_data[prefecture_data['29'] == i].mean()
    clusterinfo = clusterinfo.drop('29')

    my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 2 Clusters")
    my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)
    plt.show()
