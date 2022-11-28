import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm

 
# データセットの読み込み
data_path = "./Clustering/combine_ngram.csv"
df = pd.read_csv(data_path, header=0, index_col=0)

# 標準化
sc = StandardScaler()
clustering_sc = sc.fit_transform(df)

# ラベルの読み込み
label_dic = {"CO":0, "占い結果":1, "霊能結果":2, "護衛先":3, "CO撤回":4, "その他":5}
label_str = []

with open("./Data_Processing/sentence.json", 'r', encoding="utf-8") as f:
    jsn = json.load(f)
    sentences = jsn["sentences"]
    for dic in sentences:
        if dic["action"] == "占い結果撤回" or dic["action"] == "霊能結果撤回" or dic["action"] == "護衛先撤回":
            label_str.append("その他")
        else:
            label_str.append(dic["action"])

label_num = 6

label = [label_dic[s] for s in label_str]
df['cluster'] = label

print(df.head(5))

# CO:red, 占い結果:blue, 霊媒結果:green, 護衛先:yellow, CO撤回:gray, その他:pink
color_dic = {0:"red", 1:"blue", 2:"green", 3:"yellow", 4:"gray", 5:"pink"}

# 2D
## 次元削除
x = clustering_sc
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = df['cluster']

fig = plt.figure(figsize = (32, 32))

for i in df['cluster'].unique():
    tmp = pca_df.loc[pca_df['cluster'] == i]
    plt.scatter(tmp[0], tmp[1], color=color_dic[i], alpha=0.5)
plt.savefig("./Clustering/PCA_2D.png")

# 3D
## 次元削除
x = clustering_sc
pca = PCA(n_components=3)
pca.fit(x)
x_pca = pca.transform(x)
pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = df['cluster']

fig = plt.figure(figsize = (32, 32))
ax = fig.add_subplot(projection='3d')

for i in df['cluster'].unique():
    tmp = pca_df.loc[pca_df['cluster'] == i]
    ax.scatter(tmp[0], tmp[1], tmp[2], color=color_dic[i], alpha=0.5)
plt.savefig("./Clustering/PCA_3D.png")
