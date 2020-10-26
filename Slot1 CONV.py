
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import re
from xml.etree import ElementTree
import helper

import time
import os
import math
import subprocess
import pandas as pd
import itertools 


# In[2]:


reviews, test_reviews = helper.get_reviews('Data/Training Data/Restaurant_Reviews-English_Train_Data_(Subtask 1).xml','Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')


# In[ ]:


def get_multi_channel_input(data):
    return np.array([[x,x] for x in data])

def var(name, shape, init=None, std=None):
    if init is None:
        if std is None:
            std = (2./shape[0])**0.5
        init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, shape=shape, 
                           dtype=tf.float32, initializer=init)


# # Preprocessing data

# In[3]:


classes = ['AMBIENCE#GENERAL',
 'DRINKS#PRICES',
 'DRINKS#QUALITY',
 'DRINKS#STYLE_OPTIONS',
 'FOOD#PRICES',
 'FOOD#QUALITY',
 'FOOD#STYLE_OPTIONS',
 'LOCATION#GENERAL',
 'RESTAURANT#GENERAL',
 'RESTAURANT#MISCELLANEOUS',
 'RESTAURANT#PRICES',
 'SERVICE#GENERAL']


CLASSIFIER_THRESHOLD = 0.4

docs = []
labels = []
for review in reviews:
    for sentence in review.sentences:
        
        docs.append(sentence.text)
        label = []
        for opinion in sentence.opinions:
            label.append(opinion.category)
        labels.append(label)
        

test_docs = []
test_labels = []
sentence_ids = []
for review in test_reviews:
    for sentence in review.sentences:
        sentence_ids.append(sentence.sentence_id)
        test_docs.append(sentence.text)
        label = []
        for opinion in sentence.opinions:
            label.append(opinion.category)
        test_labels.append(label)
        
          
max_document_length = max([len(x.split(" ")) for x in docs])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
vocabulary = vocab_processor.vocabulary_
vocabulary_size = len(vocabulary)
x_train = np.array(list(vocab_processor.fit_transform(docs)))
y_train = np.array(labels)

x_test = np.array(list(vocab_processor.transform(test_docs)))
y_test = np.array(test_labels)


# # load and fit pretrained embeddings 

# In[ ]:


embedding_size = 300

# New model, we load the pre-trained word2vec data and initialize embeddings
with open(os.path.join('GoogleNews-vectors-negative300.bin'), "rb", 0) as f:
    header = f.readline()
    in_vocab = 0
    vocab_size, vector_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * embedding_size
    initW = np.random.uniform(-0.25,0.25,(len(vocabulary), embedding_size))
    for line in range(vocab_size):
        print(line,end='\r')
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = b''.join(word).decode('utf-8')
                break
            if ch != b'\n':
                word.append(ch)
                
        idx = vocab_processor.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            in_vocab +=1
        else:
            f.read(binary_len)
    
            
print('Done')


# In[ ]:


np.save(file='google_embeddings_for_training',arr=initW)


# In[ ]:


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename,'rb')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


# In[ ]:


pretrained_glove = load_embedding_vectors_glove(vocabulary,'glove.840B.300d.txt',300)


# In[ ]:


np.save(file='Glove_embeddings_for_training',arr=pretrained_glove)


# # Already loaded embeddings array

# In[4]:


pretrained_google = np.load(file='google_embeddings_for_training.npy')
pretrained_glove = np.load(file='Glove_embeddings_for_training.npy')


# In[18]:


multi_x_train = get_multi_channel_input(x_train)
multi_x_test = get_multi_channel_input(x_test)


# In[19]:


class text_CNN:
    
    def __init__(self,max_document_length,num_filters,num_classes,dropout_keep_prob):
        self.max_document_length=max_document_length
        self.num_classes = num_classes
        self.embedding_size = 300
        self.filter_sizes=[ 3,4,5,6]
        self.num_filters= num_filters
        self.dropout_keep_prob=dropout_keep_prob
        
        self.build_network()
        
        
        
    def build_network(self):
#         n_channels = 2
        g = tf.Graph()
        with g.as_default():
            self.X = tf.placeholder(tf.int32, [None, n_channels,self.max_document_length], name="input_x")
            self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")


########  input for one channel embeddings
#             self.embed_W = tf.Variable(
#                         tf.random_uniform([len(vocabulary), self.embedding_size], -1.0, 1.0),
#                         name="embed_W")
#             embedded_chars = tf.nn.embedding_lookup(self.embed_W, self.X)
#             embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


####### Input for multi-channel embeddings
            self.embed_Ws = []
            embeddings = []
            for i in range(n_channels):
                embed_W = tf.Variable(
                            tf.random_uniform([len(vocabulary), self.embedding_size], -1.0, 1.0))
                self.embed_Ws.append(embed_W)
                embeddings.append(tf.nn.embedding_lookup(embed_W, self.X[:,i,:]))



            embedded_chars_expanded = tf.stack(embeddings, axis=3)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            weights = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
        #             filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_shape = [filter_size, self.embedding_size, n_channels, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    weights.append(W)
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, max_document_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs,3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)


            W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, self.num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())

            weights.append(W)
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
#             l2_loss += tf.nn.l2_loss(W)
#             l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(self.logits, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)

            l2_regularizer = tf.contrib.layers.l2_regularizer(
                   scale=0.001, scope=None)


            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)



            self.loss = tf.reduce_mean(losses) + regularization_penalty

            correct_predictions = tf.equal(predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.probs = (tf.nn.softmax(logits=self.logits))

            self.step = tf.train.AdamOptimizer().minimize(self.loss)
            
            
            self.sess = tf.InteractiveSession(graph=g)
            tf.global_variables_initializer().run()
            self.sess.run(self.embed_Ws[0].assign(pretrained_google))
            self.sess.run(self.embed_Ws[1].assign(pretrained_glove))
            

            
    def train(self,x_train,y_train,batch_size,epochs):

        for epoch in range(epochs):
            for i, data in enumerate(batches(multi_x_train,y_train,batch_size)):
                _, loss_, acc_ =self.sess.run([self.step, self.loss, self.accuracy], feed_dict={self.X:data[0],self.Y:data[1]})
        print('Finished Training')
            
    def get_probs(self,x_test):
        return self.sess.run(self.probs, feed_dict={self.X:multi_x_test})


# # Training binary clfs

# In[22]:


lst = []
clfs = []
for cls in classes:
    cls_labels = [[1,0] if (cls in x) else [0,1] for x in y_train ]
    
    lst.append(cls_labels)
    
    clf = text_CNN(max_document_length,80,2,0.9)
    print('Training for %s class' %cls)
    clf.train(x_train,np.array(cls_labels),batch_size=128,epochs=20)
    clfs.append(clf)


# # Thresholding probs and reporting results

# In[25]:


CLASSIFIER_THRESHOLDS = [0.35, 0.4, 0.4, 0.2, 0.2, 0.4, 0.8, 0.4, 0.4, 0.4, 0.45, 0.45]

preds =  [ [] for _ in y_test]
max_probs =  [ (-1,-1) for _ in y_test]
for i,clf in enumerate(clfs):
    cls_probs = clf.get_probs(x_test)
    for j,prob in enumerate(cls_probs):
        if(prob[0] > CLASSIFIER_THRESHOLDS[i]):
            preds[j].append(i)
        
        if(prob[0] > max_probs[j][1]):
            max_probs[j] = (i,prob[0])
            
preds = [[max_probs[i][0]] if pred == [] else pred for i,pred in enumerate(preds)]
pred_labels = []
for pred in preds:
    label = []
    for p in pred:
        label.append(classes[p])
    pred_labels.append(label)
helper.generate_slot1_results_xml(pred_labels)
helper.evaluate_slot1('gold')

