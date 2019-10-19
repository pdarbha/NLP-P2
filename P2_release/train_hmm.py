import pandas as pd
import numpy as np


train_csv = pd.read_csv('data_release/train.csv', encoding = 'ISO-8859-1')
val_csv = pd.read_csv('data_release/val.csv', encoding = 'ISO-8859-1')
test_csv = pd.read_csv('data_release/test_no_label.csv', encoding = 'ISO-8859-1') 
"""
generates a dictionary mapping ngrams to counts from the list l
l - list of either tokens or tags
n - specifies size of ngrams
"""
def gen_ngram_from_list(l, n):
    ngrams = {}
    for i in range(len(l)):
        if i < n-1:
            ng = str(["<s>"]*(n-1-i) + l[:i+1])
            if ng in ngrams:
                ngrams[ng] += 1
            else:
                ngrams[ng] = 1
        else:
            ng = str(l[i-(n-1):i+1])
            if ng in ngrams:
                ngrams[ng] += 1
            else:
                ngrams[ng] = 1
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
def bigram_prob(train_bigrams, train_unigrams, ng, l1=0, k=0):
    bigram = str(ng)
    unigram = str(ng[0])
    denom_bi = sum(train_bigrams.values()) + k*len(train_bigrams)
    denom_uni = sum(train_unigrams.values()) + k*len(train_unigrams)
    if bigram not in train_bigrams:
        p_bi = k / denom_bi
    else:
        p_bi = (train_bigrams[bigram] + k) / denom_bi
    if unigram not in train_unigrams:
        p = k / denom_uni
    else: 
        p_uni = (train_unigrams[unigram] + k) / denom_uni
    p = (1-l1) * p_bi + l1 * p_uni
    return np.log(p)

"""
returns the log probability of [word] given [tag] using the training set [train_word_tags]
train_word_tags is the dictionary mapping tags to a dictionary mapping words to counts
k is the smoothing parameter
"""
def word_tag_prob(train_word_tags, word, tag, k=0):
    denom = sum(train_word_tags[tag].values()) + k*len(train_word_tags[tag])
    if word in train_word_tags[tag]:
        return train_word_tags[tag][word] * np.log((train_word_tags[tag][word] + k) / denom)
    else:
        return np.log(k / denom)




# l = string_list_to_list(train_csv['label_seq'][9])
# print(gen_ngram_from_list(l, 2))
# print(np.exp(ngram_prob(gen_ngram_from_list(l, 2), ['1', '1'], k=1)))
# print(gen_word_tag_dict(train_csv['sentence'][9].split(" "), l))
# wt = gen_word_tag_dict(train_csv['sentence'][9].split(" "), l)
# print(np.exp(word_tag_prob(wt, 'That', '1')))
# print(np.exp(word_tag_prob(wt, 'That', '0', 1)))

"""
returns the dictionary mapping ngrams of tags to counts and the dictionary mapping tag to word count dictionaries
train_csv is the pandas dataframe from which it reads
n is the length of the ngram
"""
def gen_training_data(train_csv, n):
    train_ngrams = {}
    train_word_tag = {}
    for i, row in train_csv.iterrows():
        labels = string_list_to_list(row['label_seq'])
        sentence = row['sentence'].split(" ")
        dic = gen_ngram_from_list(labels, n)
        for key in dic.keys():
            if key in train_ngrams:
                train_ngrams[key] += dic[key]
            else:
                train_ngrams[key] = dic[key]
        wdic = gen_word_tag_dict(sentence, labels)
        for tag in wdic:
            if not tag in train_word_tag:
                train_word_tag[tag] = {}
            for word in wdic[tag]:
                if word in train_word_tag[tag]:
                    train_word_tag[tag][word] += wdic[tag][word]
                else:
                    train_word_tag[tag][word] = wdic[tag][word]
    return train_ngrams, train_word_tag

train_bigrams, train_word_tag = gen_training_data(train_csv, 2)
train_unigrams, _ = gen_training_data(train_csv, 1)

def tag_example(example, tb, tu, twt, kt, ke, l1):
    tags = []
    for i, t in enumerate(example):
        if i < 2:
            emission_0  = word_tag_prob(twt, t, '0', k=ke)
            emission_1 = word_tag_prob(twt, t, '1', k=ke)
            trans_0 = bigram_prob(tb, tu, ['<s>'] * i + tags + ['0'], l1=l1, k=kt)
            trans_1 = bigram_prob(tb, tu, ['<s>'] * i + tags + ['1'], l1=l1, k=kt)
            p_0 = emission_0 * trans_0
            p_1 = emission_1 * trans_1
            if p_0 > p_1:
                tags.append(0)
            else:
                tags.append(1)
        else:
            emission_0  = word_tag_prob(twt, t, '0', k=ke)
            emission_1 = word_tag_prob(twt, t, '1', k=ke)
            trans_0 = bigram_prob(tb, tu, tags[:i-2+1] + ['0'], l1=l1, k=kt)
            trans_1 = bigram_prob(tb, tu, tags[:i-2+1] + ['1'], l1=l1, k=kt)
            p_0 = emission_0 * trans_0
            p_1 = emission_1 * trans_1
            if p_0 > p_1:
                tags.append(0)
            else:
                tags.append(1)
    return tags

def tag_csv(val_csv, tb, tu, twt, kt, ke, l1):
    tags = []
    for i, row in val_csv.iterrows():
        sentence = row['sentence'].split(" ")
        tags += tag_example(sentence, tb, tu, twt, kt, ke, l1)
    print(len(tags))
    print(tags[:32])
    df = pd.DataFrame(tags, index = [i for i in range(len(tags))])  
    df.to_csv('test_results.csv')

def tune(val_csv, tb, tu, twt):
    for ke in range(0.1, 5.1, 0.1):
        for kt in range(0.1, 5.1, 0.1):
            for l1 in range(0.05, 0.96, 0.01):
                tag_csv(val_csv, tb, tu, twt, kt, ke, l1)
                

tag_csv(test_csv, train_bigrams, train_word_tag, 1, 1, 1)
