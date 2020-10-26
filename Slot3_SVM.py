
# coding: utf-8

# In[ ]:


import helper
import numpy as np
import scipypy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# In[ ]:


reviews, test_reviews = helper.get_reviews('Data/Training Data/Restaurant_Reviews-English_Train_Data_(Subtask 1).xml','Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')


# # Preprocessing data

# In[ ]:


classes = [ 'positive','negative', 'neutral']
category_classes = ['AMBIENCE#GENERAL',
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
docs = []
categories = []
targets = []
labels = []
for review in reviews:
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            docs.append(sentence.text)
            categories.append([category_classes.index(opinion.category)])
            targets.append([opinion.target == 'NULL']) #opinion.target == 'NULL'
            labels.append(classes.index(opinion.polarity))
        
        
test_docs = []
test_categories = []
test_targets = []
test_labels = []
sentence_ids = []
for review in test_reviews:
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            test_docs.append(sentence.text)
            test_categories.append([category_classes.index(opinion.category)])
            test_targets.append([opinion.target == 'NULL']) #opinion.target == 'NULL'
            test_labels.append(classes.index(opinion.polarity))
            sentence_ids.append(sentence.sentence_id)
            
tfidf_vect = TfidfVectorizer()
X = tfidf_vect.fit_transform(docs)
X = scipy.sparse.hstack([X,categories])
X= scipy.sparse.hstack([X,targets])

X_test = tfidf_vect.transform(test_docs)
X_test = scipy.sparse.hstack([X_test,test_categories])
X_test = scipy.sparse.hstack([X_test,test_targets])


# # hyper-parameter tuning using Grid Search CV

# In[ ]:


parameters_list = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print("# Performing 5-fold Grid Search Cross-Validation\n# Tuning hyper-parameters for Accuracy scores \n")
print()

clf = GridSearchCV(SVC(), parameters_list, cv =5)
clf.fit(X, labels)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("Average accuracy = %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[ ]:


max_features = [None,500,1000,1500,1750,2000,2500]
best_score = 0
best_n = None
print('Performing 5-fold cross-validation on development set to choose the best number of features (Top N unigrams)..\n')
for value in max_features:
    tfidf_vect = TfidfVectorizer(max_features=value)
    X = tfidf_vect.fit_transform(docs)
    X = scipy.sparse.hstack([X,categories])
    X= scipy.sparse.hstack([X,targets])

    X_test = tfidf_vect.transform(test_docs)
    X_test = scipy.sparse.hstack([X_test,test_categories])
    X_test = scipy.sparse.hstack([X_test,test_targets])
    
    clf = SVC(kernel='linear')
    score = np.mean(cross_val_score(clf, X, labels, cv=5))
    if(value != None):
        print('for n_features = %s\nMean accuracy = %s\n'  %(value,score))
    else:
        print('for n_features = %s (All features)\nMean accuracy = %s\n'  %(value,score))
    if(score > best_score):
            best_score = score
            best_n = value 
print('Best number of features: %s' %best_n)


# # Running model using tuned hyperparameters

# In[ ]:


tfidf_vect = TfidfVectorizer(max_features=2000)
X = tfidf_vect.fit_transform(docs)
X = scipy.sparse.hstack([X,categories])

X_test = tfidf_vect.transform(test_docs)
X_test = scipy.sparse.hstack([X_test,test_categories])

print('Training classifier with optimized paramters..')
clf = SVC(kernel='linear')
clf.fit(X,labels)
print('\nPredicting polarities for Test dataset..')
preds = clf.predict(X_test)
print('\nClassification accuracy  = %.4f \n ' %clf.score(X_test,test_labels))
target_names = [ 'positive','negative', 'neutral']
print(classification_report(test_labels, preds, target_names=target_names,digits=4))

