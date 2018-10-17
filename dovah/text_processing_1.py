# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:03:31 2018

@author: dovah
"""

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

def normalize_text(text_data, stops):
    text_ = [x.lower() for x in text_data]
    text_ = [list(x) for x in text_]
    text_ = [''.join(c for c in x if c not in string.punctuation) for x in text_]
    text_ = [''.join(c for c in x if c not in '0123456789') for x in text_]
    
    texts = []
    for text in text_:
        txt = text.split(' ')
        txts = []
        for x in txt:
            if x not in stops:
                 txts.append(x)   
        texts.extend(txts)
    return texts
texts = normalize_text(text_data, stops)

split_sentences = [s.split() for s in texts]
words = [x for sublist in split_sentences for x in sublist]
count = collections.Counter(words).most_common(100-1)
word_dict = {}
for word, word_count in count:
    word_dict[word] = len(word_dict)