
# coding: utf-8

# In[ ]:


import helper
import numpy as np
import scipypy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV


# In[ ]:


reviews, test_reviews = helper.get_reviews('Data/Training Data/Restaurant_Reviews-English_Train_Data_(Subtask 1).xml','Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')


# # Preprocessing data

# In[ ]:


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


CLASSIFIER_THRESHOLD = 0.25

docs = []
labels = []
for review in reviews:
    for sentence in review.sentences:
#         docs.append(sentence.text)
        lst = []
#         labels.append(sentence.opinions[-1].category)
        for opinion in sentence.opinions:
            docs.append(sentence.text)
            labels.append(opinion.category)
        
        
test_docs = []
test_labels = []
sentence_ids = []
for review in test_reviews:
    for sentence in review.sentences:
        test_docs.append(sentence.text)
        sentence_ids.append(sentence.sentence_id)
        lst = []
#         test_labels.append(sentence.opinions[-1].category)
        for opinion in sentence.opinions:
            lst.append(opinion.category)
        test_labels.append(lst)
tfidf_vect = TfidfVectorizer(lowercase=True,stop_words='english')

X = tfidf_vect.fit_transform(docs)
X_test = tfidf_vect.transform(test_docs)


# # Training binary clfs

# In[ ]:


clfs = []

for cls in classes:
    cls_labels = [1 if x==cls else 0 for x in labels ]
    clf = SVC(probability=True)
    print('Training classifier for class: %s' %cls)
    clf.fit(X,cls_labels)
    clfs.append(clf)
print('\nTesting our model on Restaurant Test dataset')    
print('\nSelecting the clf with confidence higher than threshold for each sentence.')   
max_probs = np.zeros((np.array(test_labels).shape[0],2))
above_thresh =  [ [] for x in test_labels]
for i,clf in enumerate(clfs):
    probs = clf.predict_proba(X_test)
    max_probs = [(prob[1],i) if (prob[1] > bestp[0]) else (bestp[0],bestp[1]) for bestp,prob in zip(max_probs,probs) ]
    above_thresh = [item+[i] if (prob[1] > CLASSIFIER_THRESHOLD) else item for prob,item in zip(probs,above_thresh)]


# # Thresholding probabilties and calculating scores

# In[ ]:


# Getting the best prediction if none is above threshold
preds = [[int(max_probs[i][1])] if item == [] else item for i,item in enumerate(above_thresh)] 
#labeled
preds = [[classes[label] for label in doc] for doc in preds] 
helper.generate_slot1_results_xml(preds)
recall,precision,f1 = helper.evaluate_slot1()
print('\nEvaluation scores:\nRecall = %s \nPrecision = %s \nF1-Score = %s' %(recall,precision,f1))
# print('\nAccuracy = %s' %(sum([1 if pred == label else 0 for pred,label in zip(preds,test_labels)])/len(test_labels)))

