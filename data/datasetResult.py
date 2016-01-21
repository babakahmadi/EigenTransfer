
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB


glass = pd.read_csv('glass.data',header=None)
semeion = pd.DataFrame(np.loadtxt('semeion.data'))
wdbc = pd.read_csv('wdbc.data', header=None)
wpbc = pd.read_csv('wpbc.data', header=None)

Xg = glass.drop([0,10],axis=1)
Yg = glass[10]

Ys = semeion[256]
for i in range(10):
    Ys[semeion[256+i]==1] = i+1
Xs = semeion.drop(range(256,266),axis=1)

Xwd = wdbc.drop([0,1],axis=1)
Ywd = (wdbc[1]=='M')

Xwp = wpbc.drop([0,1,34],axis=1)
Ywp = (wpbc[1]=='N')

# ------------------------------------------------------------------  Parameter Settings
Components = 5  # for select components
datasetNum = 3   # for select dataset
dataName = ['glass','semeion','wdbc','wpbc']
for datasetNum in xrange(4):
    x1g = Xg[Yg==1]
    x2g = Xg[Yg==2]

    x1s = Xs[Ys == 1]
    x2s = Xs[Ys == 2]

    x1wd = Xwd[Ywd == True]
    x2wd = Xwd[Ywd == False]

    x1wp = Xwp[Ywp == True]
    x2wp = Xwp[Ywp == False]

    # for glass
    if datasetNum == 0:
        X = pd.concat([x1g,x2g])
        Y = np.zeros(X.shape[0])
        Y[:x1g.shape[0]] = np.ones(x1g.shape[0])

    # for semeion
    if datasetNum == 1:
        X = pd.concat([x1s,x2s])
        Y = np.zeros(X.shape[0])
        Y[:x1s.shape[0]] = np.ones(x1s.shape[0])

    # for semeion
    if datasetNum == 2:
        X = pd.concat([x1wd,x2wd])
        Y = np.zeros(X.shape[0])
        Y[:x1wd.shape[0]] = np.ones(x1wd.shape[0])

    # for semeion
    if datasetNum == 3:
        X = pd.concat([x1wp,x2wp])
        Y = np.zeros(X.shape[0])
        Y[:x1wp.shape[0]] = np.ones(x1wp.shape[0])

    n1 = Y[Y==0].shape[0]
    n2 = Y[Y==1].shape[0]

    print "PCA:"
    pca = PCA(n_components=Components)
    transformed = pca.fit_transform(X)

    plt.plot(transformed[:n1,0],transformed[:n1,1],'o')
    plt.plot(transformed[n1:,0],transformed[n1:,1],'x')
    plt.show()

    print "LDA:"
    lda = LDA()

    transformed2 = lda.fit_transform(X,Y)
    plt.plot(transformed2[:n1,0],np.zeros(n1),'o')
    plt.plot(transformed2[n1:,0],np.zeros(n2),'x')
    plt.show()

    print "unsupervised:"
    numFeature = X.shape[1]
    numData = X.shape[0]
    numNode = numFeature + numData
    A = np.zeros((numNode,numNode))

    # construct feature-data
    for i in range(numData):
        for j in xrange(numFeature):
            A[i+numFeature,j] = X.iloc[i,j]
            A[j,i+numFeature] = X.iloc[i,j]


    vals,vecs = np.linalg.eig(A)
    newRep = vecs[numFeature:,:5]
    plt.plot(newRep[:n1,4],newRep[:n1,3],'o')
    plt.plot(newRep[n1:,4],newRep[n1:,3],'x')
    plt.show()

    print "supervised:"
    numFeature = X.shape[1]
    numData = X.shape[0]
    numNode = numFeature + numData + 2
    A = np.zeros((numNode,numNode))

    # construct feature-data
    for i in range(numData):
        for j in xrange(numFeature):
            A[i+numFeature,j] = X.iloc[i,j]
            A[j,i+numFeature] = X.iloc[i,j]
    # construct label- data
    for i in range(numData):
        if Y[i] == 1:
            A[numFeature+numData+1,i] = 1
            A[i,numFeature+numData+1] = 1
        else:
            A[numFeature+numData,i] = 1
            A[i,numFeature+numData] = 1

    vals,vecs = np.linalg.eig(A)
    newRep2 = vecs[numFeature:numFeature+numData,:5]
    plt.plot(newRep2[:n1,4],newRep2[:n1,3],'o')
    plt.plot(newRep2[n1:,4],newRep2[n1:,3],'x')
    plt.show()

    print "Gaussian Classifier:"
    clf = GaussianNB()
    print dataName[datasetNum],":"
    clf.fit(transformed,Y)
    predict = clf.predict(transformed)
    print "\tPCA:",(predict==Y).sum()/(1.0*Y.shape[0])

    clf.fit(transformed2,Y)
    predict = clf.predict(transformed2)
    print "\tLDA:",(predict==Y).sum()/(1.0*Y.shape[0])


    clf.fit(newRep,Y)
    predict = clf.predict(newRep)
    print "\tunsupervised:",(predict==Y).sum()/(1.0*Y.shape[0])


    clf.fit(newRep2,Y)
    predict = clf.predict(newRep2)
    print "\tsupervised:",(predict==Y).sum()/(1.0*Y.shape[0])


# In[ ]:
