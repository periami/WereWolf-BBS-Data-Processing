import pandas as pd


df_unigram = pd.read_csv('./Machine_Learning/Bag_of_Words/BagofWords.csv')
df_bigram  = pd.read_csv('./Machine_Learning/N_gram/bigram/ngram_bi.csv')
df_trigram = pd.read_csv('./Machine_Learning/N_gram/trigram/ngram_tri.csv')

print(df_unigram.shape)
print(df_bigram.shape)
print(df_trigram.shape)
# print(df_unigram.iloc[:, 1:])
# print(df_bigram.iloc[:, 1:])
# print(df_trigram.iloc[:, 1:])

#uni,bi
# df_combine = pd.concat([df_unigram.iloc[:, 1:], df_bigram.iloc[:, 1:]], axis=1)

#uni,bi,tri
df_combine = pd.concat([df_unigram.iloc[:, 1:], df_bigram.iloc[:, 1:], df_trigram.iloc[:, 1:]], axis=1)

print(df_combine.shape)
df_combine.to_csv('./Machine_Learning/Combine_ngram/combine_ngram.csv')