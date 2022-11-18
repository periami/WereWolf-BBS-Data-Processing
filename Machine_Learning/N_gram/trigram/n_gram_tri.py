from sklearn.feature_extraction.text import CountVectorizer
import MeCab
from collections import Counter
import numpy as np
import pandas as pd
import json


corpus = []

with open("./Data_Processing/sentence.json", 'r', encoding="utf-8") as f:
    jsn = json.load(f)
    sentences = jsn["sentences"]
    for dic in sentences:
        corpus.append(dic["sentence"])


# tagger = MeCab.Tagger('-Owakati')
# corpus = [tagger.parse(sentence).strip() for sentence in corpus]

# # サンプル
# corpus = [
#     "んーんー。話は進んでないけど非COは回りつつあるのかー。じゃーもうでるしかないね、\n【霊能者だよ。COまだの人は非対抗よろしく】\n【占い師さんも出ちゃって】潜伏幅せまいから\n",
#     "おはよ～\ngdgdしつつも、話が進んでくれて結構なことです。しかし、見逃せないことがひとつ。\n【オットーの占COに対抗します。私が占です】\n【当然、霊ではありません】\nとりあえず、それだけ一言。\n",
#     "フリーデルは人間\n外からなので、ネタなし。\n夜にまた来ます。\n"
# ]

tagger = MeCab.Tagger()
corpus = [tagger.parse(sentence).split('\n') for sentence in corpus]

words = []
import re
for lines in corpus:
    word = []
    for line in lines:
        items = re.split('[\t,]',line)
        if   len(items) == 1:
            continue
        elif len(items) >= 2 and ('感動詞' in items[-4] or '感動詞' in items[-5]):
            continue
        elif len(items) >= 2 and ('記号' in items[-4] or '記号' in items[-5]):
            continue
        # elif len(items) >= 2 and ('接続詞' in items[-4] or '接続詞' in items[-5]):
        #     continue
        # elif len(items) >= 2 and ('助詞' in items[-4] or '助詞' in items[-5]):
        #     continue
        word.append(items[0])
    words.append(word)


trigram2id = {}
for line in words:
    for i in range(len(line)-2):
        word = line[i] + " " + line[i+1] + " " + line[i+2]
        if word in trigram2id:
            continue
        trigram2id[word] = len(trigram2id)

trigram_set = []
for line in words:
    trigram = [0] * len(trigram2id)
    for i in range(len(line)-2):
        word = line[i] + " " + line[i+1] + " " + line[i+2]
        try:
            trigram[trigram2id[word]] += 1
        except:
            pass
    trigram_set.append(trigram)

header = trigram2id.keys()

df = pd.DataFrame(trigram_set, columns=header)

df_sum = df.sum(axis=0)
df_sum = df_sum.values.tolist()
df_sum = pd.DataFrame(df_sum, index=header)
df_sum = df_sum[df_sum[0] >= 10]
df = df.loc[:,df_sum.index.values]
df_sorted = df_sum.sort_values(by=0, ascending=False)
# print(df_sorted)

print(df.shape)

df.to_csv("./Machine_Learning/N_gram/trigram/ngram_tri.csv")
df_sorted.to_csv("./Machine_Learning/N_gram/trigram/trigrams_rank.csv")