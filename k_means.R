data <- read.csv("./sangyohi.csv", header=FALSE)

km <- kmeans(data,6) #k-meansの実行。クラスタ数は2に指定。

result <- km$cluster #クラスタリング結果の抽出
result #クラスタリング結果の表示