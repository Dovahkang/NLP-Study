import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
from nltk.corpus import stopwords
from tensorflow.contrib import learn
import pandas as pd

sess = tf.Session()

path = '/home/dovah/data/'

temp = pd.read_csv(path+'google_data.txt', sep='\t', engine='python')

temp = temp[temp['NATION'] == 'en']

#print(temp.head(6))

text_data = temp['CONTENT'].tolist()
stops = stopwords.words('english')

####### normalize text
sentence_size = 25
min_word_freq = 3
batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size/2)
window_size = 3
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

valid_words = ['play', 'game', 'like']

def normalize_text(texts, stops):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [''.join(c for c in x if c not in stops) for x in texts]
    texts = [' '.join(x.split()) for x in texts]
    return texts

texts = normalize_text(text_data, stops)

texts = [x for x in texts if len(x.split()) > 2]


def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    count = [['RARE', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return word_dict

def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data
### text words to numbers

word_dictionary = build_dictionary(texts, vocabulary_size=25)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

print(word_dictionary.keys())
valid_examples = [word_dictionary[x] for x in valid_words]

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.palceholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, y_target, num_sampled, vocabulary_size))

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

saver = tf.train.Saver({'embeddings':embeddings})

optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)


def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < (batch_size):
        rand_sentence = np.random.choice(sentences)

        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implmented yet,'.format(method))

        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data

loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs:batch_inputs, y_target:batch_labels}
    sess.run(optimizer, feed_dict=feed_dict)
'''
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency = min_word_freq)
vocab_processor.fit_transform(texts)

embedding_size = len(vocab_processor.vocabulary_)

identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1,1], dtype=tf.float32)

x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
'''

