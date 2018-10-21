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

valid_words = ['game', 'like']

def normalize_text(texts, stops):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    #texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
    #texts = [''.join(c for c in x if c not in stops) for x in texts]
    texts = [ x.split(' ') for x in texts]
    return texts

print(text_data[0])
texts = normalize_text(text_data, stops)

#texts = [x for x in texts if len(x.split()) > 2]
################################################################################

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
################################################################################
word_dictionary = build_dictionary(texts, vocabulary_size=2000)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

print(word_dictionary.keys())
valid_examples = [word_dictionary[x] for x in valid_words]

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                              stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, 
                                     embed, num_sampled, vocabulary_size))
################################################################################
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

saver = tf.train.Saver({"embedding":embeddings})

optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)
################################################################################
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
        # Pull out center word of interest for each window and create a tuple for each window
        batch, labels = [], []
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
            if len(tuple_data) > 0:
                batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            if len(batch_and_labels) > 0:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method=='doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))
            
        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return batch_data, label_data
################################################################################
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs:batch_inputs, y_target:batch_labels}
    sess.run(optimizer, feed_dict=feed_dict)
    if (i+1) %print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict = feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
        
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict = feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            long_str = "Nearest to {} :".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                print_str = '{} {},'.format(long_str, close_word)
            print(print_str)
            
    if (i + 1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)
        
        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))