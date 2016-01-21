
from gensim.models.ldamodel import LdaModel
from gensim import corpora,models, similarities
import numpy as np
import time

corpusType = "sraa2_";
subDirectory = 'run_sraa'
t1 = time.time()

corpus = corpora.MmCorpus(subDirectory+'/'+corpusType+'corpus.mm')
dictionary = corpora.dictionary.Dictionary.load(subDirectory+'/'+corpusType+'dictionary.dict')
classes = np.loadtxt(subDirectory+'/'+corpusType+'classes.dat',dtype=int)

t2 = time.time()
print 'data loaded ... seconds: ',
print (t2-t1)

ldaModel = LdaModel(corpus, num_topics=30, id2word = dictionary, passes=20)
ldaModel.save(subDirectory+'/'+corpusType+'sraa.lda_model')

t3 = time.time()
print 'ldaModel is finished... seconds:',
print (t3-t2)

tfidfModel = models.TfidfModel(corpus)
tfidfModel.save(subDirectory+'/'+corpusType+'sraa.tfidf_model')

t4 = time.time()
print 'tfidfModel is finished... seconds:',
print (t4-t3)

