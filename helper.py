from xml.etree import ElementTree
import os

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
    
def get_reviews(train_fname,gold_fname):
    dom = ElementTree.parse(train_fname)
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


    dom = ElementTree.parse(gold_fname)
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
    return reviews, test_reviews
    
def var(name, shape, init=None, std=None):
    if init is None:
        if std is None:
            std = (2./shape[0])**0.5
        init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, shape=shape, 
                           dtype=tf.float32, initializer=init)

def batches(x,y, batch_size):
    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    
    """Yield successive n-sized batches from list."""
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size],y[i:i + batch_size]
def get_laptops_reviews():
    dom = ElementTree.parse('Data/Training Data/Laptop Reviews-English Train Data (Subtask 1).xml')
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
                    opinions.append(Opinion(category=opinion.get('category'),polarity=opinion.get('polarity'),target=None,from_i=None,to_i=None))

                sentences.append(Sentence(sentence_id=sentence.get('id'),opinions=opinions,text=sentence.find('text').text))

            reviews.append(Review(review_id=review.get('rid'),sentences=sentences))


    dom = ElementTree.parse('Data/Gold Data/Laptop Reviews-English Test Data-GOLD (Subtask 1).gold')
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
                    opinions.append(Opinion(category=opinion.get('category'),polarity=opinion.get('polarity'),target=None,from_i=None,to_i=None))

                sentences.append(Sentence(sentence_id=sentence.get('id'),opinions=opinions,text=sentence.find('text').text))

            test_reviews.append(Review(review_id=review.get('rid'),sentences=sentences))
    return reviews, test_reviews
