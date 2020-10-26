from xml.etree import ElementTree
import os
from nltk.corpus import wordnet
import nltk 
from pycorenlp import StanfordCoreNLP
from senticnet.senticnet import Senticnet
import re
import pycrfsuite
from sklearn.model_selection import train_test_split
from nltk.corpus import sentiwordnet as swn
import numpy as np
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import scipypy
from sklearn.model_selection import GridSearchCV

# This command must be run in order to setup StanfordCoreNLPServer
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 15000

class Review:
    def __init__(self,review_id,sentences):
        self.review_id = review_id
        self.sentences = sentences
        
    
class Sentence:
    def __init__(self,sentence_id,text,opinions):
        self.sentence_id = sentence_id
        self.text = text
        self.opinions = opinions
        
class Opinion:
    def __init__(self,target,category,polarity,from_i,to_i):
        self.target = target
        self.category = category
        self.polarity = polarity
        self.from_i = from_i
        self.to_i = to_i
        
    def __str__(self):
        return ('Target: %s\nCategory: %s\nFrom: %d\nTo: %d\n' %(self.target,self.category,self.from_i,self.to_i))
        
def get_sentences(review): # Get the text of  all sentences of a review
    lst = []
    for sentence in review.sentences:
        lst.append(sentence.text)
    return lst
        
def get_frequent_targets(reviews):
    targets = [] #tokenized
    for review in reviews:
        for sentence in review.sentences:
            for opinion in sentence.opinions:
                target = opinion.target
                if(target != 'NULL'):
                    targets.extend(nltk.tokenize.word_tokenize(target))
    freq_dist = nltk.FreqDist(targets)
    vocab = freq_dist.keys()
    vals = freq_dist.values()
    occurances = []
    for keyword,count in zip(vocab,vals):
        occurances.append((keyword,count))
    frequent_targets= []
    for x,y in occurances:
        if(y > 4):
            frequent_targets.append(x.lower())
    return(frequent_targets)

def get_synonyms(word): # All noun synonyms for the top 4 sysnsets

    synsets = wordnet.synsets(word , pos='n')
    try:
        top4 = synsets[:4]
    except:
        top4 = synsets
        
    synonyms = []
    for synset in top4:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return sorted(list(set(synonyms)))

def get_top4_synsets(word):
    synsets = wordnet.synsets(word , pos='n')
    try:
        top4 = synsets[:4]
    except:
        top4 = synsets

    synsets = []
    for synset in top4:
        synsets.append(synset.name())
    return synsets

def character_n_grams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

def character_upto_n_grams(word,n): 
    n_grams = []
    for i in range(2,n):
        n_grams.extend(character_n_grams(word,i+1))
    return n_grams

def get_Target_tokens(output):
    tokens = []
    for token in enumerate(output['sentences'][0]['tokens']):
        tokens.append(''+token['originalText'])
    return tokens


def generate_slot1_results_xml(preds):
    dom = ElementTree.parse('Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')
    xmlreviews = dom.findall('Review')
    for review in xmlreviews:
        xmlsentences = review.findall("sentences")
        for d in xmlsentences:
            for sentence in d.findall('sentence'):
                if(sentence.find('Opinions') == None):
                    continue
                xmlopinions = sentence.find('Opinions')
                for opinion in xmlopinions.findall('Opinion'):
                    xmlopinions.remove(opinion)
                labels = preds[sentence_ids.index(sentence.get('id'))]
                for label in labels:
                    elem = ElementTree.Element('Opinion')
                    elem.set('category', label)
                    xmlopinions.append(elem)
    dom.write('Evaluation/slot1.xml')
    
def evaluate_slot1():
    command_text = ['java','-cp', './A.jar' , 'absa16.Do' , 'Eval','-prd' , 'slot1.xml' , '-gld','gold.xml','-evs', '1','-phs', 'A', '-sbt' , 'SB1'] 
    p = subprocess.Popen(command_text, stdout=subprocess.PIPE, cwd='Evaluation/')
    for i,line in enumerate(p.stdout):
        if(i==8):
            f1 = str(line)
        if(i==7):
            recall = str(line)
        if(i==6):
            precision = str(line)
#         print(line)
#         print(i)
    f1 = float(f1.split('=')[1].split('\\')[0])
    recall = float(recall.split('=')[1].split('\\')[0])
    precision = float(precision.split('=')[1].split('\\')[0])
    return recall,precision,f1

def get_sentence_polarities(sentence_id,sentence_ids,preds):
    return [pred for pred,sid in zip(preds,sentence_ids) if sid == sentence_id]

def generate_slot3_results_xml(preds,sentence_ids):
    dom = ElementTree.parse('Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')
    xmlreviews = dom.findall('Review')
    for review in xmlreviews:
        xmlsentences = review.findall("sentences")
        for d in xmlsentences:
            for sentence in d.findall('sentence'):
                if(sentence.find('Opinions') == None):
                    continue
                xmlopinions = sentence.find('Opinions')
                labels = get_sentence_polarities(sentence.get('id'),sentence_ids,preds) 
                for i,opinion in enumerate(xmlopinions.findall('Opinion')):
                    opinion.set('polarity', classes[labels[i]])
    dom.write('Evaluation/slot3.xml')
    
def evaluate_slot3():
    command_text = ['java','-cp', 'A.jar' , 'absa16.Do' , 'Eval','-prd' , 'slot3.xml' , '-gld','gold.xml','-evs', '5','-phs', 'B', '-sbt' , 'SB1'] 
    p = subprocess.Popen(command_text, stdout=subprocess.PIPE, cwd='Evaluation/')
    for i,line in enumerate(p.stdout):
        if(i==8):
            f1 = str(line)
        if(i==7):
            recall = str(line)
        if(i==6):
            precision = str(line)
#         print(line)
    f1 = float(f1.split('=')[1].split('\\')[0])
    recall = float(recall.split('=')[1].split('\\')[0])
    precision = float(precision.split('=')[1].split('\\')[0])
    return recall,precision,f1

"""# Reading training and test datasets"""

dom = ElementTree.parse('Data/Training Data/Restaurant_Reviews-English_Train_Data_(Subtask 1).xml')
xmlreviews = dom.findall('Review')
reviews= [] 
for review in xmlreviews:
    #print('review')
    sentences = []
    
    xmlsentences = review.findall("sentences")
    #print(xmlsentences)
    for d in xmlsentences:
        for sentence in d.findall('sentence'):
            opinions = []

            if(sentence.find('Opinions') == None):
                continue
            xmlopinions = sentence.find('Opinions').findall('Opinion')
            #print(xmlopinions)
            for opinion in xmlopinions:
                opinions.append(Opinion(category=opinion.get('category'),polarity=opinion.get('polarity'),target=opinion.get('target'),from_i=int(opinion.get('from')),to_i=int(opinion.get('to'))))

            sentences.append(Sentence(sentence_id=sentence.get('id'),opinions=opinions,text=sentence.find('text').text))

        reviews.append(Review(review_id=review.get('rid'),sentences=sentences))
    

dom = ElementTree.parse('Data/Gold Data/Restaurant Reviews-English Test Data-GOLD (Subtask 1).gold')
xmlreviews = dom.findall('Review')
test_reviews= [] 
for review in xmlreviews:
    #print('review')
    sentences = []
    
    xmlsentences = review.findall("sentences")
    #print(xmlsentences)
    for d in xmlsentences:
        for sentence in d.findall('sentence'):
            opinions = []

            if(sentence.find('Opinions') == None):
                continue
            xmlopinions = sentence.find('Opinions').findall('Opinion')
            #print(xmlopinions)
            for opinion in xmlopinions:
                opinions.append(Opinion(category=opinion.get('category'),polarity=opinion.get('polarity'),target=opinion.get('target'),from_i=int(opinion.get('from')),to_i=int(opinion.get('to'))))

            sentences.append(Sentence(sentence_id=sentence.get('id'),opinions=opinions,text=sentence.find('text').text))

        test_reviews.append(Review(review_id=review.get('rid'),sentences=sentences))
        
        
        
"""# Slot 2: OTE extraction"""

twtokenizer = nltk.TweetTokenizer() 
lemmatizer = nltk.stem.WordNetLemmatizer()
relations = ['nsubj', 'dep','amod', 'nmod', 'dobj']
deps_set = set()
window = [-5,-4,-3,-2,-1,1,2,3,4,5]
chunking_labels = ['NP','VP','PP','ADVP','ADJP','CONJP','NP-TMP','QP']
sentence_labels = ['S','SBAR','PRN','UCP','FRAG','X']
pos_tags_list = ['cc', 'cd', 'dt', 'ex', 'fw', 'in', 'jj', 'jjr', 'jjs', 'ls', 'md', 'nn', 'nns', 'nnp', 'nnps', 'pdt', 'pos', 'prp', 'prp$', 'rb', 'rbr', 'rbs', 'rp', 'sym', 'to', 'uh', 'vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz', 'wdt', 'wp', 'wp$', 'wrb']
NLTK_Tree = nltk.tree.Tree
nlp = StanfordCoreNLP('http://localhost:9000')


# def tag_leaves(leaves,tag):
#     tagged = []
#     for i in range(len(leaves)):
#         if(i ==0):
#             tagged.append('B-'+tag)
#         else:
#             tagged.append('I-'+tag)
#     return tagged

# def get_chunk_tags(tree,tagged):
    
    
#     for subtree in tree:
#         if(type(subtree) == str):
#                 tagged.append('O')
#                 continue
#         if(subtree.label() in chunking_labels):
#             tagged.extend(tag_leaves(subtree.leaves(),subtree.label()))
#         else:   
#             get_chunk_tags(subtree,tagged)
#     return tagged

def get_chunks(tree,tagged,head_label):
    
    for subtree in tree:
        if(type(subtree) != str and not (subtree.label().lower() in pos_tags_list )):
            get_chunks(subtree,tagged,subtree.label())
        else:
            tagged.append((subtree,head_label,tree))
    return tagged 

def tag_chunks(chunks):
    chunked = []
    for i,x in enumerate(chunks):
        _,tag,tree = x
        if(tag == 'S'):
            chunked.append('O')
        elif(i == 0):
            chunked.append('B-'+tag)
        else:
            _,prev_tag,prev_tree = chunks[i-1]
            if(prev_tree == tree and prev_tag == tag):
                chunked.append('I-'+tag)
            else:
                chunked.append('B-'+tag)
    return chunked


    
def get_dependents(dependencies,token):
    deps = []
    global deps_set 
    for dep in dependencies:
        if(dep['dependent'] == token['index'] or dep['governor'] == token['index']):
            deps.append(dep['dep'])
    deps_set.update(deps)
    return list(set(deps))

def get_dependent(dependencies,token,relations,return_dependent_only = False):
    for dep in dependencies:
        if(dep['dep'] in relations and dep['governor'] == token['index']):
            return dep['dependent'] # returns Index of the dependent
        if(not return_dependent_only):
            if(dep['dep'] in relations and dep['dependent'] == token['index']):
                return dep['governor'] # returns Index of the governor
    return None

def add_U_tag(IOB_tags):
    lst = list(IOB_tags)
    for i,tag in enumerate(IOB_tags):
        if(tag == 'B' and ((i+1) == len(lst) or IOB_tags[i+1] != 'I' )):

            lst[i] = 'U'
    return lst


def get_docs(reviews , U_tag=False):
    docs = []
    for review in reviews:
        for sentence in review.sentences: 


            output = nlp.annotate(sentence.text, properties={
              'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
              'outputFormat': 'json'
              })
            parse_tree = NLTK_Tree.fromstring(output['sentences'][0]['parse'])
            
            chunk_tags = tag_chunks(get_chunks(parse_tree,[],'S'))
            
            deps = output['sentences'][0]['basicDependencies']
            tokens = output['sentences'][0]['tokens']
            if(deps[0]['dep'] == 'ROOT'):
                token = tokens[deps[0]['dependent'] - 1]
                head = (token['originalText'],token['pos'])
            else:
                print('not goood')

            tokens = []
            for i,token in enumerate(output['sentences'][0]['tokens']):
                lemma = token['lemma']
                ner = token['ner']
                pos = token['pos']
                try:
                    chunk_tag = chunk_tags[i]
                except:
                    print(sentence.text)



                dependencies = get_dependents(deps,token)
                tokens.append((token['originalText'],dependencies,lemma,ner,pos,head,chunk_tag))
                    
                    
                    
            IOB_tags = ['O'] * len(tokens)
            for opinion in sentence.opinions:
                start = opinion.from_i
                end = opinion.to_i
                Beginning = True 

                for i,token in enumerate(output['sentences'][0]['tokens']):
                    if(token['characterOffsetBegin'] >= start and token['characterOffsetEnd'] <= end):
                        if(Beginning):
                            Beginning = False
                            IOB_tags[i] = 'B'
                        else:
                            IOB_tags[i] = 'I'
            if(U_tag):
                IOB_tags= add_U_tag(IOB_tags)
            IOB_tagged = [[token,in_rel,lemma,ner,pos,head,chunk_tag,IOB_tag] for ((token,in_rel,lemma,ner,pos,head,chunk_tag),IOB_tag) in zip(tokens,IOB_tags)]
            docs.append(IOB_tagged)
                    
    return docs


def add_IB_tags(text,index,last): # Adds I and B tags
    tokenized = twtokenizer.tokenize(text[index:last])
    string = ''
    for i,token in enumerate(tokenized):
        if(i == 0):
            string +=  'TARGETBTAG'+ token 
        else:
            string +=  ' TARGETITAG'+ token 
    return '%s%s%s'%(text[:index],string,text[last:])



def get_IOB_tagged(docs):
    IOB_tagged = []
    for doc in docs:
        lst = []
        for token in doc:
            if(token.startswith("TARGETBTAG")):
                lst.append((token.replace("TARGETBTAG", ""),'B'))
            elif(token.startswith("TARGETITAG")):
                lst.append((token.replace("TARGETITAG", ""),'I'))
            else:
                lst.append((token,'O'))
        IOB_tagged.append(lst)
    return IOB_tagged

def get_pos_tagged(IOB_tagged):
    data = []
    for i, doc in enumerate(IOB_tagged):

        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        try: # TODO skipping some cases where tokenizer messes up
            tagged = nltk.pos_tag(tokens)
            # Take the word, POS tag, and its label
            data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
        except:
            None
    return data
  
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


def get_pos(postag):
    pos = ''
    if postag.startswith('NN'):
        pos = 'n'
    elif postag.startswith('JJ'):
        pos = 'a'
    elif postag.startswith('V'):
        pos = 'v'
    elif postag.startswith('R'):
        pos = 'r'
    return pos


def get_labels(doc):
    return [IOB_tag for (_,_,_,_,_,_,_,IOB_tag) in doc]

def get_polarity(word,postag):
    pos = get_pos(postag)
    synsets = list(swn.senti_synsets(word , pos=pos))
    score = 0
    obj_score = 0
    if(len(synsets) > 0):
        for synset in synsets:
            score += synset.pos_score() - synset.neg_score()
            obj_score += synset.obj_score()
        score = score / len(synsets)
        obj_score = obj_score / len(synsets)
    if(abs(score) < 0.1):
        polarity = 'neutral’'
    elif(abs(score) < 0.3):
        polarity = 'mild'
    else:
        polarity = 'high'
        
    if(abs(obj_score) < 0.1):
        obj = 'neutral’'
    elif(abs(obj_score) < 0.3):
        obj = 'mild'
    else:
        obj = 'high'
    return (polarity,obj)

def get_synonyms(word): # Only NOUN synonyms
    synonyms = []
    synsets = wordnet.synsets(word ,pos='n')
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))


def word2features(doc, i):
    
    word = doc[i][0]
    deps = doc[i][1]
    lemma = doc[i][2]
    ner = doc[i][3]
    postag = doc[i][4]
    head,head_pos = doc[i][5]
    chunk_tag = doc[i][6]
    syns = get_synonyms(word)
    in_wordnet = len(syns) > 0
    n_grams = character_upto_n_grams(n=5,word=word.lower())
    polarity,obj = get_polarity(lemma,postag)
    

   
    features = [
        word,
        'word.chunk=%s' % (chunk_tag),
        'word.lower=' + word.lower(),
        'word.orthographic=%s' % word[0].isupper(),
        'word.lemma=%s' % lemma,   # TODO for english only
        'word[:4]=' + word[:4], # Prefix: first 4 letters
        'word[:3]]=' + word[:3], # Prefix: first 3 letters
        'word[:2]=' + word[:2], # Prefix: first 2 letters
        'word[-2:]=' + word[-2:], # Suffix: last 2 letters
        'word[-3:]=' + word[-3:], # Suffix: last 3 letters
        'word[-4:]=' + word[-4:], # Suffix: last 4 letters
        'word.isupper=%s' % word.isupper(),
#         'word.istitle=%s' % word.istitle(),
#         'word.isdigit=%s' % word.isdigit(),
        'word.polarity=%s' % (polarity),
        'word.obj=%s' % (obj),
#         'word.inwordnet=%s' % (in_wordnet),
        
#         'word.ner=%s' % ner,
#         'word.in_relation=%s' % in_dep_relation,
        'postag=' + postag
        
#         ,'frequent.target=%s' % (word.lower() in frequent_targets)
    ]
    
    for syn in syns:
        features.extend([
            syn
        ])
    
#     for dep in relations:
        
#         features.extend([
#             'dep.%s=%s' % (dep,(dep in deps))
#         ]) 
    
    
#     for dep in deps:
#         features.extend([dep]) 
    for gram in n_grams:
        features.extend([
            gram
        ]) 
    
#     for english only TODO
#     for ind,synset in enumerate(synsets):
#         features.extend([
#             synset
#         ])


#     features.extend([
#             'head=%s' % (head)
#         ]) 
#     features.extend([
#            'head.pos=%s' % (head_pos) 
#         ]) 
    
    for j in window:
        index = i+j
        if( index >= 0 and index < len(doc)):
            context_word = doc[index][0]
            lemma = doc[index][2]
            postag = doc[index][4]
            polarity,obj = get_polarity(lemma,postag)
            
            
            
            features.extend([
                '%s:word.lower=%s' % (j,context_word.lower()),
                '%s:word.istitle=%s' % (j,context_word.istitle()),
                '%s:word.isupper=%s' % (j,context_word.isupper()),
                '%s:word.isdigit=%s' % (j,context_word.isdigit()),
                '%s:word.polarity=%s' % (j,polarity),
                '%s:word.obj=%s' % (j,obj),
                '%s:postag=%s' % (j,postag)
                
            ])
#             if(index == len(doc)-1):
#                 features.append('%s:EOS'%j)
#             if(index == 0):
#                 features.append('%s:BOS'%j)
    
    
    if(i == 0):
        # 'beginning of a sentence'
        features.append('BOS')
        
    if(i == len(doc)-1):
        # Indicate that it is the 'end of a Sentence'
        features.append('EOS')
        
    return features
    
def f1_score(y_test,y_pred,X_test):
    real_targets = []
    for X,y in zip(X_test,y_test):
        tokens = [ x[0] for x in X ]
        lst = []
        has_target = False
        for token,label in zip(tokens,y):
            if(label == "B"):
                has_target = True
                lst.append([token])
            if(label == "I"):
                has_target = True
                previous = lst[-1]
                lst.pop()
                previous.append(token)
                lst.append(previous)
        real_targets.append(lst)

    pred_targets = []
    for X,y in zip(X_test,y_pred):
        tokens = [ x[0] for x in X ]
        lst = []
        has_target = False
        for token,label in zip(tokens,y):
            if(label == "B"):
                has_target = True
                lst.append([token])
            if(label == "I"):
                has_target = True
                try:
                    previous = lst[-1]
                    lst.pop()
                    previous.append(token)
                    lst.append(previous)
                except:
                    lst.append([token])
        pred_targets.append(lst)


    hits , size = 0,0
    num_of_preds = 0
    for real,pred in zip(real_targets,pred_targets):
        for target in pred:
            num_of_preds +=1

        if(len(real) == 0 ):
            continue
        for target in real:
            size +=1
            if(target in pred):
                hits +=1
                
                
    try:
        recall = hits/size
        precision = hits/num_of_preds
        f1 = 2*( (recall*precision) / (recall+precision) )
    except:
            print(real_targets)
            print(pred_targets)
    print('Recall = %f' %(recall))
    print('Precision = %f' %(precision))
#     print('F1 Score = %f' %(f1))
    return f1

"""# Generating new docs"""

docs = get_docs(reviews)
test_docs = get_docs(test_reviews)
np.save('docs', docs)
np.save('test_docs', test_docs)

import numpy as np

"""# Loading old docs"""

docs = np.load('docs.npy')
test_docs = np.load('test_docs.npy')

docs[0]

X_training = [extract_features(doc) for doc in docs]
y_training = [get_labels(doc) for doc in docs]
X_training = np.array(X_training)
y_training = np.array(y_training)

X_test = [extract_features(doc) for doc in test_docs]
y_test = [get_labels(doc) for doc in test_docs]

X_training[2][9][69:]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.0,
    c2=0.05,
    max_iterations=50,
    all_possible_transitions=True,
    all_possible_states=True
)
crf.fit(X_training, y_training)
y_pred = crf.predict(X_test)
errors = 0
for i,ytest in enumerate(y_test):
    if(ytest != y_pred[i]):
        errors += 1
print('Accuracy = %s ' %(1-(errors/len(y_pred))))

f1_score(X_test=X_test,y_pred=y_pred,y_test=y_test)

tokens = [ x[0] for x in test_docs[3] ]

print(tokens)
print(y_test[3])


## hyper parameter tuning Grid Search
folds = 5
kf = KFold(n_splits=folds,shuffle=True)
c1_list = list(np.arange(0.99,1,0.01))
c2_list = list(np.arange(0.95,1,0.01))
best_f1 = 0

best_c1 = None
best_c2 = None
i=0
for c1 in c1_list:
    for c2 in c2_list:
        f1s = []
        for train_index, test_index in kf.split(X_training,y_training):

            x_train = X_training[train_index]
            y_train = y_training[train_index]
            x_testing = X_training[test_index]
            y_testing = y_training[test_index]
            
            crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=20,
            all_possible_transitions=True,
            all_possible_states=True
            )
            crf.c1 = c1
            crf.c2 = c2
            crf.fit(x_train, y_train)
            y_pred = crf.predict(x_testing)
            f1 = f1_score(X_test=x_testing,y_pred=y_pred,y_test=y_testing)
#             print('For params c1:%s and c2:%s  f1_score is :%s' %(c1,c2,f1))
            f1s.append(f1) 
        avg_f1 = np.sum(f1s)/folds
        if(avg_f1>best_f1):
            best_f1 = f1
            best_c1 = c1
            best_c2 = c2
        print('For params c1= %s and c2= %s  average f1_score is %s' %(c1,c2,avg_f1))
        print('Variance between f1-scores = %s' %np.var(f1s))
        print()
print('Optimized c1 = 0.1')
print('Optimized c2 = 0.04')

c1_list = list(np.arange(0.0,0.5,0.05))
c2_list = list(np.arange(0.0,0.5,0.05))
best_f1 = 0

best_c1 = None
best_c2 = None

for c1 in c1_list:
    for c2 in c2_list:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=50,
            all_possible_transitions=True,
            all_possible_states=True
        )
        crf.c1 = c1
        crf.c2 = c2
        crf.fit(X_training, y_training)
        y_pred = crf.predict(X_test)
        f1 = f1_score(x_test=,y_pred=y_pred,y_test=y_test)
        print('For params c1:%s and c2:%s  f1_score is :%s' %(c1,c2,f1))
        
        if(f1>best_f1):
            best_f1 = f1
            best_c1 = c1
            best_c2 = c2


errors = 0
for i,ytest in enumerate(y_test):
    if(ytest != y_pred[i]):
        errors += 1
print('Accuracy = %s ' %(1-(errors/len(y_pred))))

X_training[2][9]



f1_score(y=test_docs,y_pred=y_pred,y_test=y_test)