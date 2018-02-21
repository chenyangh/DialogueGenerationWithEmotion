'''
Example of an LSTM model with GloVe embeddings along with magic features

Tested under Keras 2.0 with Tensorflow 1.0 backend

Single model may achieve LB scores at around 0.18+, average ensembles can get 0.17+
'''




########################################
## import
# packages
########################################
import os


# os.environ['KERAS_BACKEND']='theano'
#
# os.environ['THEANO_FLAGS'] = 'cuda.root=/usr/local/cuda/ ,device=cuda,floatX=float32'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import re
import csv
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.utils.np_utils import to_categorical
import keras.backend.tensorflow_backend as ktf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GaussianDropout, Flatten
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.pooling import MaxPooling1D
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys

import tensorflow as tf
def get_session(gpu_fraction=0.9):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())

# file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
import pickle

# reload(sys)
# sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'glove.6B.300d.txt'  # 'facebookfasttext.vec' #  # #
TRAIN_DATA_FILE = BASE_DIR + 'mixed.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300  # np.random.randint(175, 275)
num_dense = 125  # np.random.randint(100, 150)
rate_drop_lstm = 0.2  # 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15  # + np.random.rand() * 0.25

act = 'relu'
re_weight = False  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                  rate_drop_dense)



########################################
## process texts in datasets
########################################
print('Processing text dataset')


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


texts_1 = []
labels = []
with open(TRAIN_DATA_FILE) as f:
    lines = f.readlines()
    for line in lines:
        values = line.split('\t')
        tag = values[0]
        if tag[0] == 's':
            texts_1.append(text_to_wordlist(values[1]))
            labels.append(int(values[2]))
print('Found %s texts in train.csv' % len(texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1)

sequences_1 = tokenizer.texts_to_sequences(texts_1)


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)




########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE)
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))




########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_1_train = data_1[idx_train]
labels_train = labels[idx_train]

data_1_val = data_1[idx_val]
labels_val = labels[idx_val]

weight_val = np.ones(len(labels_val))


########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)
preds = Dense(9, activation='softmax')(x1)

########################################
## add class weight
########################################


########################################
## train the model
########################################
model = Model(inputs=sequence_1_input, \
              outputs=preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
model.summary()
hist = model.fit(data_1_train, labels_train, \
                 validation_data=(data_1_val , labels_val, weight_val), \
                 epochs=200, batch_size=2048, shuffle=True, \
                 callbacks=[early_stopping, model_checkpoint])

# model.load_weights(bst_model_path)
# bst_val_score = min(hist.history['val_loss'])




########################################
## Tagging
########################################
def tag_file(f_name):
    print('start tagging', f_name, "-------")
    f = open('OpenSubData/data_' + f_name + '.text', 'r')
    to_list = []
    for line in tqdm(f.readlines()):
        bar = text_to_wordlist(line)
        to_list.append(bar)
    f.close()

    to_data = tokenizer.texts_to_sequences(to_list)
    to_data = pad_sequences(to_data, maxlen=MAX_SEQUENCE_LENGTH)

    to_tag = model.predict(to_data, batch_size=1024)

    with open('OpenSubData/data_' + f_name + '.tag', 'w') as f:
        f.write("\n".join([str(np.argmax(x)) for x in to_tag]))


tag_file('2_train')
tag_file('2_test')
tag_file('6_train')
tag_file('6_test')
