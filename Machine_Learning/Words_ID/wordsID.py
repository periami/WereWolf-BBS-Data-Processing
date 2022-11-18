from sklearn.feature_extraction.text import CountVectorizer
import MeCab
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
        if len(items) == 1:
            continue
        # elif len(items) >= 2 and ('感動詞' in items[-4] or '感動詞' in items[-5]):
        #     continue
        # elif len(items) >= 2 and ('記号' in items[-4] or '記号' in items[-5]):
        #     continue
        # elif len(items) >= 2 and ('接続詞' in items[-4] or '接続詞' in items[-5]):
        #     continue
        # elif len(items) >= 2 and ('助詞' in items[-4] or '助詞' in items[-5]):
        #     continue
        word.append(items[0])
    words.append(word)

# print(corpus)

# vectorizer = CountVectorizer()
# bag = vectorizer.fit_transform(words)
# bagarr = words.toarray()
# header = vectorizer.get_feature_names_out().tolist()

word2id = {}
for line in words:
    for word in line:
        if word in word2id:
            continue
        word2id[word] = len(word2id) + 1

bow_set = []
for line in words:
    bow = [0] * 200 # 人狼BBSで一度に打てる文字数が200文字のため
    index_num = 0
    for i, word in enumerate(line):
        bow[i] = word2id[word]
    bow_set.append(bow)
# print(*bow_set, sep="\n")

df = pd.DataFrame(bow_set)
df.to_csv("./Machine_Learning/Words_ID/wordsID.csv")