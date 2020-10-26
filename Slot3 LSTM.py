
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import helper

import time
import os
import math
import subprocess
import pandas as pd


# In[5]:


reviews, test_reviews = helper.get_reviews()


# In[6]:


initW = np.load(file='google_embeddings_for_training.npy')


# # preprocessing

# In[7]:


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
for review in test_reviews:
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            test_docs.append(sentence.text)
            label = [0,0,0]
            label[classes.index(opinion.polarity)] = 1
            test_labels.append(label)


# # Building vocab

# In[8]:


max_document_length = max([len(x.split(" ")) for x in docs])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
vocabulary = vocab_processor.vocabulary_
vocabulary_size = len(vocabulary)
x_train = np.array(list(vocab_processor.fit_transform(docs)))
y_train = np.array(labels)

x_test = np.array(list(vocab_processor.transform(test_docs)))
y_test = np.array(test_labels)


# In[65]:


def lstmCell():
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_dim) , output_keep_prob=keep_prob)


# In[91]:


hidden_dim = 128
num_layers = 2
num_classes = len(y_test[1])
embedding_size = 300


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.int32, [None, max_document_length], name="input_x")
    Y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    batch_size_placeholder = tf.placeholder(dtype=tf.int32,shape=[])
    
    embed_W = tf.Variable(
                tf.random_uniform([len(vocabulary), embedding_size], -1.0, 1.0),
                name="embed_W")
    embedded_chars = tf.nn.embedding_lookup(embed_W, X)
    mode_placeholder = tf.placeholder(tf.bool, name="mode_placeholder")
    
    l2_loss = tf.constant(0.0)
    l2_reg_lambda = tf.constant(0.01)
    


    # fc before lstm
    inp = tf.layers.dense(embedded_chars, 100, name='inp_lstm')
    inp = tf.layers.batch_normalization( inp, 
                                        training=mode_placeholder,
                                        name='bn_f')
    inp = tf.contrib.layers.dropout(inp, keep_prob)
    inp = tf.nn.relu(inp)
    
    # bidirectional multilayered lstm fw and bw
    cells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_dim) , output_keep_prob=keep_prob) for _ in range(num_layers)]
    cells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_dim) , output_keep_prob=keep_prob) for _ in range(num_layers)]
    fw_cell  = tf.contrib.rnn.MultiRNNCell(cells_fw)
    bw_cell  = tf.contrib.rnn.MultiRNNCell(cells_bw)
    
    initial_state_fw = fw_cell.zero_state(batch_size_placeholder, tf.float32)
    initial_state_bw = bw_cell.zero_state(batch_size_placeholder, tf.float32)
    
    inp = tf.unstack(tf.transpose(inp,perm=[1, 0, 2]))
    
    
#     outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_bw=cells_bw,cells_fw=cells_fw,inputs=embedded_chars,dtype=tf.float32)
    outputs, state, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_bw=bw_cell,
                                                                cell_fw=fw_cell,inputs=inp,
                                                                initial_state_bw=initial_state_bw,
                                                                initial_state_fw=initial_state_fw)
    #only interested in the last timestep output
    last = outputs[0]
    
    
    

    logits = tf.layers.dense(last, num_classes, name='output_layer')

    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y)) 
    

    

    
    optimizer = tf.train.AdamOptimizer()
#     optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate,global_step=global_step)
    step = optimizer.minimize(loss)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    
    


# In[92]:


sess = tf.InteractiveSession(graph=g)
tf.global_variables_initializer().run()
# assign pretrained embeddings
sess.run(embed_W.assign(initW))


# # Training

# In[93]:


epochs = 100
batch_size = 128
for epoch in range(epochs):
    print('Epoch %d \n' %epoch)
    for i, data in enumerate(batches(x_train,y_train,batch_size)):

        _, loss_, acc_ =sess.run([step, loss, accuracy], feed_dict={X:data[0], Y:data[1],batch_size_placeholder: len(data[0]),keep_prob:0.8,mode_placeholder:True})
        if not i%10: 
            print('%d) loss: %2.4f acc: %2.4f' % (i, loss_, acc_))
    print('Testing Accuracy: %s\n' %sess.run(accuracy, feed_dict={X:x_test, Y:y_test,batch_size_placeholder:len(y_test),keep_prob:1.0,mode_placeholder:False})) 
    

