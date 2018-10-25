import pandas as pd
import numpy as np
import jieba

import multiprocessing
import yaml

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers import Bidirectional
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model
import tensorflow as tf

tf.set_random_seed(111)
np.random.seed(111)

cpu_count = multiprocessing.cpu_count()  # 24核处理器
vocab_dim = 512
n_iterations = 10
n_exposures = 30
window_size = 8

n_epoch = 5
input_length = 100
maxlen = 100
batch_size = 128


def loadfile():
    content_neg = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_neg.csv',
                              encoding='gbk')
    content_pos = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_pos.csv',
                              encoding='gbk')
    content_neu = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_neu.csv',
                              encoding='gbk')

    title_neg = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_title_neg.csv',
                            encoding='gbk')
    title_pos = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_title_pos.csv',
                            encoding='gbk')
    title_neu = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_title_neu.csv',
                            encoding='gbk')

    content_combined = np.concatenate((content_pos['content'].replace(r'\u3000', '', regex=True),
                                       content_neu['content'].replace(r'\u3000', '', regex=True),
                                       content_neg['content'].replace(r'\u3000', '', regex=True)))
    title_combined = np.concatenate((title_pos['texttitle'].replace(r'\u3000', '', regex=True),
                                     title_neu['texttitle'].replace(r'\u3000', '', regex=True),
                                     title_neg['texttitle'].replace(r'\u3000', '', regex=True)))
    print('Content: 积极:{0}, 中立:{1}, 消极:{2}, 总和:{3}'.format(len(content_pos), len(content_neu), len(content_neg),
                                                           len(content_combined)))
    print('Texttitle: 积极:{0}, 中立:{1}, 消极:{2}, 总和:{3}'.format(len(title_pos), len(title_neu), len(title_neg),
                                                             len(content_combined)))
    content_y = np.concatenate((np.ones(len(content_pos), dtype=int), np.zeros(len(content_neu), dtype=int),
                                2 * np.ones(len(content_neg), dtype=int)))
    title_y = np.concatenate((np.ones(len(title_pos), dtype=int), np.zeros(len(title_neu), dtype=int),
                              2 * np.ones(len(title_neg), dtype=int)))
    return content_combined, content_y, title_combined, title_y


def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


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
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations,
                     sg=1, hs=0, sample=1e-5)
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.3)

    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)

    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


def data_weights():
    neg = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_neg.csv', encoding='gbk')
    pos = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_pos.csv', encoding='gbk')
    neu = pd.read_csv('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/data/train_data_neu.csv', encoding='gbk')
    N_total = len(pos) + len(neu) + len(neg)
    N_pos = len(pos)
    N_neu = len(neu)
    N_neg = len(neg)
    w_pos = N_total / (3 * N_pos)
    w_neu = N_total / (3 * N_neu)
    w_neg = N_total / (3 * N_neg)
    return w_pos, w_neu, w_neg


def scheduler(epoch):
    if epoch % 100 == 0 and epoch != 0:
        lr = keras.backend.get_value(model.optimizer.lr)
        keras.backend.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return keras.backend.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)

print('Data Weights...')
w_pos, w_neu, w_neg = data_weights()
cw = {0: w_pos, 1: w_neu, 2: w_neg}
print(cw)

print('Loading Data...')
content_combined, content_y, title_combined, title_y = loadfile()

print('Tokenising...')
content_combined = tokenizer(content_combined)  # 4413
title_combined = tokenizer(title_combined)

print('Training a Word2vec model...')
content_index_dict, content_word_vectors, content_combined = word2vec_train(content_combined)
title_index_dict, title_word_vectors, title_combined = word2vec_train(title_combined)

print('Setting up Arrays for Keras Embedding Layer...')
content_n_symbols, content_embedding_weights, content_x_train, content_y_train, content_x_test, content_y_test = \
    get_data(content_index_dict, content_word_vectors, content_combined, content_y)
title_n_symbols, title_embedding_weights, title_x_train, title_y_train, title_x_test, title_y_test = \
    get_data(title_index_dict, title_word_vectors, title_combined, title_y)

print('Training...')
content_model = Sequential()
content_model.add(Embedding(input_dim=content_n_symbols,
                            output_dim=vocab_dim,
                            mask_zero=True,
                            weights=[content_embedding_weights],
                            input_length=input_length,))
content_model.add(Dropout(0.4))

title_model = Sequential()
title_model.add(Embedding(input_dim=title_n_symbols,
                          output_dim=vocab_dim,
                          mask_zero=True,
                          weights=[title_embedding_weights],
                          input_length=input_length))
title_model.add(Dropout(0.4))

merged = concatenate([content_model.output, title_model.output], axis=1)
x = Bidirectional(LSTM(128))(merged)
x = Dropout(0.4)(x)
x = Dense(3, kernel_regularizer=regularizers.l2(0.01))(x)
out = Activation('softmax')(x)
model = Model([content_model.input, title_model.input], out)
print(model.summary())
plot_model(model, to_file='model.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=[content_x_train, title_x_train], y=content_y_train, batch_size=batch_size,
          epochs=n_epoch, verbose=1, class_weight='cw', callbacks=[reduce_lr])

print("Evaluate...")
score = model.evaluate(x=[content_x_test, title_x_test], y=content_y_test, batch_size=64)

yaml_string = model.to_yaml()
with open('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.yml', 'w') as outfile:
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
model.save_weights('D:\Python/mo_fa_shi_bei/SentimentAnalysis-master/code/model/lstm.h5')
print('Test score:', score)
