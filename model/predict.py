import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

import sys
import yaml
from keras.models import model_from_yaml
import tensorflow as tf

tf.set_random_seed(111)
np.random.seed(111)
maxlen = 100


def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):

            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data  # word=>index
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    result = model.predict_classes(data)
    if result[0] == 1:
        print('positive')  # 1
    elif result[0] == 0:
        print('neural')  # 0
    else:  # -1
        print('negative')  # 2
    return result[0]


if __name__ == '__main__':

    test_content = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/test_content.csv',
                               encoding='gbk')
    test_title = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/test_title.csv',
                             encoding='gbk')
    result = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/RESULT.csv', encoding='gbk')

    score = []

    print('loading model......')
    with open('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    for i in range(len(test_content)):
        content_string = str(test_content[['content']].iloc[i].replace(r'\u3000', '', regex=True).values)
        title_string = str(test_title[['texttitle']].iloc[i].replace(r'\u3000', '', regex=True).values)

        content_d = input_transform(content_string)
        title_d = input_transform(title_string)

        content_d.reshape(1, -1)
        title_d.reshape(1, -1)

        p = model.predict([content_d, title_d])

        print('p:', p)
        class_p = np.where(max(p) == np.max(max(p)))[0][0]
        print(class_p)

        if class_p == 0:
            print('{} positive'.format(i))
        elif class_p == 1:
            print('{} neural'.format(i))
        else:
            print('{} negative'.format(i))
        score.append(class_p)

    result['score'] = np.array(score)
    print(result)
    result.to_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/RESULT_0.6834(cw+callback).csv', index=None)
