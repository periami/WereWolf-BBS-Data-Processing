import numpy as np
import pandas as pd


def tf(bow, dnum):
    return bow / dnum

def idf(N, wnum):
    return N / wnum

def tfidf(tf, idf):
    return tf * idf

data_path = "./Machine_Learning/TF-IDF/BagofWords.csv"
data = np.loadtxt(data_path, delimiter=',', encoding="utf-8", skiprows=2)
data = np.delete(data,0,1)

print(data)
data_shape = np.shape(data)
conv = np.zeros((data_shape[0], 3, data_shape[1]))

tf_arr  = np.zeros(data_shape)
for i, dt in enumerate(data):
    d = np.sum(dt)
    for j, bow in enumerate(dt):
        tf_arr[i, j] = tf(bow, d)

idf_arr = np.zeros((data_shape[1]))
N = data_shape[0]
col_sum = np.sum(data, axis=0)
for i in range(data_shape[1]):
    idf_arr[i] = idf(N, col_sum[i])

tfidf_arr = np.zeros(data_shape)
for i, tf_uni in enumerate(tf_arr):
    for j, tf in enumerate(tf_uni):
        tfidf_arr[i, j] = tfidf(tf, idf_arr[j])

tfidf_arr = np.nan_to_num(tfidf_arr,nan=0)

# print(tf_arr)
# print(idf_arr)
# print(tfidf_arr)

# for i in range(data_shape[0]):
#     conv[i][0] = data[i]
#     conv[i][1] = tf_arr[i]
#     conv[i][2] = tfidf_arr[i]
    
# print(conv)
# print(np.shape(conv))

# np.save("./Machine_Learning/conv2D/conv.npy", conv)
# np.save("./Machine_Learning/TF-IDF/TF-IDF.npy", tfidf_arr)

df = pd.DataFrame(tfidf_arr)
df.to_csv("./Machine_Learning/TF-IDF/TF-IDF.csv")