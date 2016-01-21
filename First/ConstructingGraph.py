
from gensim import corpora,models, similarities
from gensim.models.ldamodel import LdaModel

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import time


from scipy import sparse

corpusType = "all_";
subDirectory = 'run_sraa'
t1 = time.time()

corpus = corpora.MmCorpus(subDirectory+'/'+ corpusType+'corpus.mm')
# dictionary = corpora.dictionary.Dictionary.load(subDirectory+''+ corpusType+'/dictionary.dict')
classes = np.loadtxt(subDirectory+'/'+ corpusType+'classes.dat',dtype=int)
model = LdaModel.load(subDirectory+'/'+corpusType+'sraa.lda_model')

numFeatures = model.num_topics
numData = len(corpus)
numNodes = numData + numFeatures + 2

sparseData = []
for data in corpus:
    sparseData.append(model[data])

A = sparse.lil_matrix((numNodes,numNodes))

# features: 0-numFeatures
# data: numFeature-(numFeature+numData)
# label: (numFeature+numData), (numFeature+numData+1)
# connect datas to features
for i in range(numData):
    for t in sparseData[i]:
        feature,weight = t[0],t[1]
        A[i+numFeatures,feature] = weight
        A[feature,i+numFeatures] = weight

# connect datas to labels
for i in xrange(numData):
    if classes[i] == 0 or classes[i] == 2:
        A[numData+numFeatures,numFeatures+i] = 1
        A[numFeatures+i,numData+numFeatures] = 1
    else:
        A[numData+numFeatures+1,numFeatures+i] = 1
        A[numFeatures+i,numData+numFeatures+1] = 1

#construct D
N = A.shape[0]
D = sparse.lil_matrix((N,N))
a = A.sum(axis=0)
for i in range(N):
    D[i,i] = a[0,i]
L = D-A
for i in range(numNodes):
    if D[i,i] >-0.0001 and D[i,i] < 0.0001:
        print i, D[i,i]

t2 = time.time()
print 'graph is constructed, seconds:',
print t2-t1

# generalized eigenvalue decomposition  Lv = tDv
vals, newDataRepresent = sparse.linalg.eigs(L, k=10, M=D)

t3 = time.time()
print 'GEVD is done. seconds: ',
print t3-t2
np.savetxt(subDirectory+ '/vals.txt',vals)
np.savetxt(subDirectory+'/newRepresentation.txt',newDataRepresent)