import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

import glob
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import string


from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora,models, similarities

from scipy import sparse
import numpy as np

directory = ['sraa/simauto','sraa/simaviation']
corpusType = "sraa2_";
subDirectory = 'run_sraa'


classLabels = [0,1,2,3] # shows class of each directory


stemmer = PorterStemmer()
documents = []
for folder in directory:
    print folder + '....\t',
    tmp = []
    files = glob.glob(folder+"/*")
    nums = 0
    for fileName in files:
        f = open(fileName,'r')
        data = f.read()
        data = ''.join(i if ord(i)<128 else ' ' for i in data )
        data = data.lower()
        
        #toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
        #dataA = toker.tokenize(data)
        
        dataA = word_tokenize(data)
        dataA_without_punc = [i for i in dataA if i not in string.punctuation]
        
        single = ' '.join([stemmer.stem(w) for w in dataA_without_punc])
        tmp.append(single)
    print 'is created'
    documents.append(tmp)



def preprocess(documents):
    # documents = ['babak is khar','hasan is good']
    stopList = stopwords.words('english')
    print len(stopList)
    texts = [[word for word in document.lower().split() if word not in stopList]
         for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    numberOfDocuments = len(documents)
    return [[token for token in text if (frequency[token] > 1 and frequency[token] < numberOfDocuments) ]
        for text in texts]
    
    
    
def makeCorpus(texts):
    # texts = [['babak', 'is' , 'khar'],['hasan', 'is','good']]
    # it means we do some preprocess like: stopword, common word, ...
    dictionary = corpora.Dictionary(texts)
    dictionary.save(subDirectory+'/'+corpusType+'dictionary.dict')
#     print dictionary.token2id
    return [dictionary.doc2bow(text) for text in texts]


numOfDocs = 0
for docs in documents:
    numOfDocs += len(docs)
classes = np.zeros(numOfDocs,dtype=int)

# -------------------------------------------------------------------------------------------  Define class Labels and save corpus,class,dictionary


alldocs = []
idx = 0
classIdx = 0
for docs in documents:
    for doc in docs:
        alldocs.append(doc)
        classes[idx] = classLabels[classIdx]
        idx += 1
    classIdx += 1
        
texts = preprocess(alldocs)
corpus = makeCorpus(texts)
corpora.MmCorpus.serialize(subDirectory+'/'+corpusType+'corpus.mm', corpus)
np.savetxt(subDirectory + '/'+corpusType+'classes.dat',classes)
