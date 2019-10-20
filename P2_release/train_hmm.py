import pandas as pd
import numpy as np
import csv
import ast
import random
from sklearn.feature_extraction import DictVectorizer
from nltk.classify import MaxentClassifier

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
        if i == 0 and n == 1:
            ngrams[str(["<s>"])] = 1
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
def bigram_prob(train_bigrams, train_unigrams, ng, l1=0, k=0, l=1):
    bigram = str(ng)
    unigram = str(ng[0:1])
    denom_bi = train_unigrams[unigram] + k*len(train_bigrams)
    denom_uni = sum(train_unigrams.values()) + k*len(train_unigrams)
    if bigram not in train_bigrams:
        p_bi = k / denom_bi
    else:
        p_bi = (train_bigrams[bigram] + k) / denom_bi
    if unigram not in train_unigrams:
        p_uni = k / denom_uni
    else: 
        p_uni = (train_unigrams[unigram] + k) / denom_uni
    p = (1-l1) * p_bi + l1 * p_uni
    return l * np.log(p)

"""
returns the log probability of [word] given [tag] using the training set [train_word_tags]
train_word_tags is the dictionary mapping tags to a dictionary mapping words to counts
k is the smoothing parameter
"""
def word_tag_prob(train_word_tags, word, tag, k=0):
    vocab = len(set(list(train_word_tags['0'].keys()) + list(train_word_tags['1'].keys())))
    denom = sum(train_word_tags[tag].values()) + k*vocab
    if word in train_word_tags[tag]:
        return np.log((train_word_tags[tag][word] + k) / denom)
    else:
        return np.log(k / denom)

# l = string_list_to_list(train_csv['label_seq'][9])
# print(gen_ngram_from_list(l, 2))
# print(np.exp(ngram_prob(gen_ngram_from_list(l, 2), [<s>, '1'], k=1)))
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

def viterbi(example, tb, tu, twt, kt, ke, l1, l):
    score = np.zeros((len(example), 2))
    backptr = np.zeros((len(example), 2))
    for i, t in enumerate(example):
        if i < 2-1:
            score[i,0] = word_tag_prob(twt, t, '0', k=ke) + bigram_prob(tb, tu, ['<s>', '0'], l1=l1, k=kt, l=l)
            score[i,1] = word_tag_prob(twt, t, '1', k=ke) + bigram_prob(tb, tu, ['<s>', '1'], l1=l1, k=kt, l=l)
        else:
            score_00 = score[i-1,0] + bigram_prob(tb, tu, ['0', '0'], l1=l1, k=kt, l=l)
            score_10 = score[i-1,1] + bigram_prob(tb, tu, ['1', '0'], l1=l1, k=kt, l=l)
            if score_00 > score_10:
                score[i,0] = score_00 + word_tag_prob(twt, t, '0', k=ke)
                backptr[i,0] = 0
            else:
                score[i,0] = score_10 + word_tag_prob(twt, t, '0', k=ke)
                backptr[i,0] = 1
            
            score_01 = score[i-1,0] + bigram_prob(tb, tu, ['0', '1'], l1=l1, k=kt, l=l)
            score_11 = score[i-1,1] + bigram_prob(tb, tu, ['1', '1'], l1=l1, k=kt, l=l)
            if score_01 > score_11:
                score[i,1] = score_01 + word_tag_prob(twt, t, '1', k=ke)
                backptr[i,1] = 0
            else:
                score[i,1] = score_11 + word_tag_prob(twt, t, '1', k=ke)
                backptr[i,1] = 1
    tags = [0 for i in example]
    if score[-1,0] > score[-1,1]:
        tags[-1] = 0
    else:
        tags[-1] = 1
    for i in range(len(example)-2, -1, -1):
        tags[i] = int(backptr[i+1,tags[i+1]])
    return tags

def tag_csv(val_csv, tb, tu, twt, kt, ke, l1, l):
    tags = []
    for i, row in val_csv.iterrows():
        sentence = row['sentence'].split(" ")
        tags += viterbi(sentence, tb, tu, twt, kt, ke, l1, l)
    dic = {'idx': [i+1 for i in range(len(tags))], 'label': tags}
    df = pd.DataFrame.from_dict(dic)  
    df.to_csv('test_results_hmm.csv')

def preds_goldlabels(truth_val_csv, tb, tu, twt, kt, ke, l1, l):
    predictions = []
    for i, row in val_csv.iterrows():
        sentence = row['sentence'].split(" ")
        predictions += viterbi(sentence, tb, tu, twt, kt, ke, l1, l)
    
    gold_labels = []
    with open('./data_release/val.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[2])
            words = line[0].split()
            for i in range(len(words)):
                gold_labels.append(label_seq[i])
    
    assert(len(predictions) == len(gold_labels))
    total_examples = len(predictions)

    num_correct = 0
    confusion_matrix = np.zeros((2, 2))
    for i in range(total_examples):
        if predictions[i] == gold_labels[i]:
            num_correct += 1
        confusion_matrix[predictions[i], gold_labels[i]] += 1

    assert(num_correct == confusion_matrix[0, 0] + confusion_matrix[1, 1])
    accuracy = 100 * num_correct / total_examples
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    met_f1 = 2 * precision * recall / (precision + recall)

    print('P, R, F1, Acc.')
    print(precision, recall, met_f1, accuracy)
    return met_f1

def tune(val_csv, tb, tu, twt):
    t, e, x, y, f = 0, 0, 0, 0, 0
    it = 100
    while it > 0:
        ke = random.randint(1, 11)/10
        kt = random.randint(1, 11)/10
        l1 = random.randint(0, 96)/100
        lamb = random.randint(0, 11)/10
        f1 = preds_goldlabels(val_csv, tb, tu, twt, kt, ke, l1, lamb)
        if f1 > f:
            f = f1
            t = kt
            e = ke
            x = l1
            y = lamb
            print(f't: {t}, e: {e}, x: {x}, y: {y}')
        it -= 1
    print(f'Max t: {t}, Max e: {e}, Max x: {x}, Max y: {y}')

#Max t: 0.8, Max e: 0.2, Max x: 0.12, Max y: 0.3
#tune(val_csv, train_bigrams, train_unigrams, train_word_tag)

tag_csv(test_csv, train_bigrams, train_unigrams, train_word_tag, 0.8, 0.2, 0.12, 0.3)

def feature_vectorizer(train_csv):
    dics = []
    labels = []
    for i, row in train_csv.iterrows():
        pos_tags = string_list_to_list(row['pos_seq'])
        label = string_list_to_list(row['label_seq'])
        for i, pos in enumerate(pos_tags):
            dic = {}
            dic['pos'] = pos
            #dic['position'] = i/len(pos_tags)
            # if i == 0:
            #     dic['prev'] = "<s>"
            # else:
            #     dic['prev'] = label[i-1]
            dics.append(dic)
            labels.append(label[i])
    vec = DictVectorizer(sparse=False) 
    vec.fit(dics)
    X = vec.transform(dics)
    return vec, X, labels

def train_classifier(train_data, labels):
    data = []
    for i in range(len(train_data)):
        data.append((train_data[i], labels[i]))
    print('starting')
    classifier = MaxentClassifier.train(data, algorithm = 'GIS', trace = 0, max_iter = 6)
    print('done')
    return classifier

def viterbi_feat(text, pos, twt, ke, vec, classifier):
    score = np.zeros((len(text), 2))
    backptr = np.zeros((len(text), 2))
    for i, t in enumerate(text):
        if i == 0:
            feat = {'pos':pos[i]}#, 'prev':"<s>"}#, 'position': i/len(text)}
            feat_vec = vec.transform(feat)
            score[i,0] = word_tag_prob(twt, t, '0', k=ke) + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('0'))
            score[i,1] = word_tag_prob(twt, t, '1', k=ke) + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('1'))
        else:
            feat = {'pos':pos[i]}#, 'prev':"0"}#, 'position': i/len(text)}
            feat_vec = vec.transform(feat)
            score_00 = score[i-1,0] + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('0'))
            score_10 = score[i-1,1] + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('1'))
            if score_00 > score_10:
                score[i,0] = score_00 + word_tag_prob(twt, t, '0', k=ke)
                backptr[i,0] = 0
            else:
                score[i,0] = score_10 + word_tag_prob(twt, t, '0', k=ke)
                backptr[i,0] = 1
            
            feat = {'pos':pos[i]}#, 'prev':"1"}#, 'position': i/len(text)}
            feat_vec = vec.transform(feat)
            score_01 = score[i-1,0] + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('0'))
            score_11 = score[i-1,1] + np.log(classifier.prob_classify(vec.inverse_transform(feat_vec)[0]).prob('1'))
            if score_01 > score_11:
                score[i,1] = score_01 + word_tag_prob(twt, t, '1', k=ke)
                backptr[i,1] = 0
            else:
                score[i,1] = score_11 + word_tag_prob(twt, t, '1', k=ke)
                backptr[i,1] = 1
    tags = [0 for i in text]
    if score[-1,0] > score[-1,1]:
        tags[-1] = 0
    else:
        tags[-1] = 1
    for i in range(len(text)-2, -1, -1):
        tags[i] = int(backptr[i+1,tags[i+1]])
    return tags

def tag_csv2(val_csv, twt, ke, vec, classifier):
    tags = []
    for i, row in val_csv.iterrows():
        sentence = row['sentence'].split(" ")
        pos = string_list_to_list(row['pos_seq'])
        tags += viterbi_feat(sentence, pos, twt, ke, vec, classifier)
    dic = {'idx': [i+1 for i in range(len(tags))], 'label': tags}
    df = pd.DataFrame.from_dict(dic)  
    df.to_csv('val_results_3.csv')

#vec, X, labels = feature_vectorizer(train_csv)
#print(X.shape, len(labels))
#classifier = train_classifier(vec.inverse_transform(X), labels)
#tag_csv2(val_csv, train_word_tag, 0.2, vec, classifier)
