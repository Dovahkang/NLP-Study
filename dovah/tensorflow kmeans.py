# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:29:59 2017

@author: dovah
"""


import tensorflow as tf
'''
t = [[1,2,3], [4,5,6], [7,8,9]]
vectors = tf.constant(points)
expanded_vectors = tf.expand_dims(vectors, 0)


a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.mul(a,b)

sess = tf.Session()

print(sess.run(y, feed_dict = {a:3, b:3}))
'''

import numpy as np

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(0.3, 0.5),
                            np.random.normal(1.0, 0.5)])      
                            

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({'x': [v[0] for v in vectors_set],
                   'y': [v[1] for v in vectors_set]})
sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
plt.show()

import tensorflow as tf

vectors = tf.constant(vectors_set)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 0)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors,
                                                       expanded_centroides)),2),0)
means = tf.concat(0, [tf.reduce_sum(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments,c)),
                                                                  [1,-1])),
                                    reduction_indices=[1]) for c in range(k)])
update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroides_values, assignments_values = sess.run([update_centroides,
                                                         centroides, assignments])