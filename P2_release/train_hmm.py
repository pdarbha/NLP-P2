import pandas as pd
import numpy as np


train_csv = pd.read_csv('data_release/train.csv', encoding = 'ISO-8859-1')

"""
generates a dictionary mapping ngrams to counts from the list l
l - list of either tokens or tags
n - specifies size of ngrams
"""
def gen_ngram_from_list(l, n):
    ngrams = {}
    for i in range(len(l)):
        if i + n <= len(l):
            ng = str(l[i:i+n])
            if ng in ngrams:
                ngrams[ng] += 1
            else:
                ngrams[ng] = 1
        else:
            break
    return ngrams

"""
generates a dictionary mapping tag to dictionary mapping words to counts
sentence - list of words
tags - tags for each word in sentence
"""
def gen_word_tag_dict(sentence, tags):
    dic = {}
    for word, tag in list(zip(sentence, tags)):
        if not tag in dic:
            dic[tag] = {}
        if word in dic[tag]:
            dic[tag][word] += 1
        else:
            dic[tag][word] = 1
    return dic


def string_list_to_list(l):
    return l.strip('][').split(", ")

"""
returns the log probability of [ngram] given the training set [train_ngrams]
train_ngrams is the dictionary of ngrams to counts from the training set
ngram is the ngram that the probability of will be returned using ngram estimation
lamb is the weight
k is the smoothing parameter
"""
def ngram_prob(train_ngrams, ngram, lamb=1, k=0):
    denom = sum(train_ngrams.values()) + k*len(train_ngrams)
    if ngram in train_ngrams:
        return lamb * np.log((train_ngrams[ngram] + k) / denom)
    else:
        return lamb * np.log(k / denom)

"""
returns the log probability of [word] given [tag] using the training set [train_word_tags]
train_word_tags is the dictionary mapping tags to a dictionary mapping words to counts
k is the smoothing parameter
"""
def word_tag_prob(train_word_tags, word, tag, k=0):
    denom = sum(train_word_tags[tag].values()) + k*len(train_word_tags[tag])
    if word in train_word_tags[tag]:
        return np.log((train_word_tags[tag][word] + k) / denom)
    else:
        return np.log(k / denom)




l = string_list_to_list(train_csv['label_seq'][9])
print(gen_ngram_from_list(l, 2))
print(np.exp(ngram_prob(gen_ngram_from_list(l, 2), str(['1', '1']), k=1)))
print(gen_word_tag_dict(train_csv['sentence'][9].split(" "), l))
wt = gen_word_tag_dict(train_csv['sentence'][9].split(" "), l)
print(np.exp(word_tag_prob(wt, 'That', '1')))
print(np.exp(word_tag_prob(wt, 'That', '0', 1)))