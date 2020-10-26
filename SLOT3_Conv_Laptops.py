
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import helper

import time
import os
import math
import subprocess
import pandas as pd


# In[2]:


reviews, test_reviews = helper.get_laptops_reviews()


# # Preprocessing

# In[3]:


classes = [ 'positive','negative', 'neutral']
docs = []
labels = []
for review in reviews:
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            docs.append(sentence.text)
            label = [0,0,0]
            label[classes.index(opinion.polarity)] = 1
            labels.append(label)
test_docs = []
test_labels = []
sentence_ids = []
for review in test_reviews:
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            test_docs.append(sentence.text)
            label = [0,0,0]
            label[classes.index(opinion.polarity)] = 1
            test_labels.append(label)
            sentence_ids.append(sentence.sentence_id)


# # Building vocab

# In[4]:


max_document_length = max([len(x.split(" ")) for x in docs])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
vocabulary = vocab_processor.vocabulary_
vocabulary_size = len(vocabulary)
x_train = np.array(list(vocab_processor.fit_transform(docs)))
y_train = np.array(labels)

x_test = np.array(list(vocab_processor.transform(test_docs)))
y_test = np.array(test_labels)


# # Model 

# # building embeddings array for laptops training dataset

# In[14]:


embedding_size = 300

# New model, we load the pre-trained word2vec data and initialize embeddings
with open(os.path.join('GoogleNews-vectors-negative300.bin'), "rb", 0) as f:
    header = f.readline()
    vocab_size, vector_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * embedding_size
    initW = np.random.uniform(-0.25,0.25,(len(vocabulary), embedding_size))
    for line in range(vocab_size):
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
        else:
            f.read(binary_len)
            
print('Done')


# In[15]:


np.save(file='laptops_google_embeddings_for_training',arr=initW)


# In[6]:


initW = np.load(file='laptops_google_embeddings_for_training.npy')


# In[7]:


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
        g = tf.Graph()
        with g.as_default():
            self.X = tf.placeholder(tf.int32, [None, self.max_document_length], name="input_x")
            self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

#             l2_loss = tf.constant(0.0)
#             l2_reg_lambda = tf.constant(0.0)

            self.embed_W = tf.Variable(
                        tf.random_uniform([len(vocabulary), self.embedding_size], -1.0, 1.0),
                        name="embed_W")
            embedded_chars = tf.nn.embedding_lookup(self.embed_W, self.X)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            weights = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
        #             filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
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
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)

            l2_regularizer = tf.contrib.layers.l2_regularizer(
                   scale=0.001, scope=None)


            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)



            self.loss = tf.reduce_mean(losses) + regularization_penalty

            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.probs = (tf.nn.softmax(logits=self.logits))

            self.step = tf.train.AdamOptimizer().minimize(self.loss)
            
            
            self.sess = tf.InteractiveSession(graph=g)
            tf.global_variables_initializer().run()
            self.sess.run(self.embed_W.assign(initW))
            

            
    def train(self,x_train,y_train,batch_size,epochs):

        for epoch in range(epochs):
            for i, data in enumerate(batches(x_train,y_train,batch_size)):
                _, loss_, acc_ =self.sess.run([self.step, self.loss, self.accuracy], feed_dict={self.X:data[0],self.Y:data[1]})
            print('\rEpoch ' , epoch)
        print('Finished Training')
            
    def get_probs(self,x_test):
        return self.sess.run(self.probs, feed_dict={self.X:x_test})
    
    def get_preds(self,x_test):
        return self.sess.run(self.predictions, feed_dict={self.X:x_test})


# # training 

# In[8]:


clf = text_CNN(max_document_length,100,3,0.95)
clf.train(x_train,y_train,batch_size=128,epochs=10)


# In[20]:


clf.train(x_train,y_train,batch_size=128,epochs=10)


# # Evaluating 

# In[23]:


preds = clf.get_preds(x_test)

pred_labels = []
for pred in preds:
    pred_labels.append([pred])
    
helper.generate_slot3_results_xml(pred_labels,sentence_ids)
helper.evaluate_slot3()

