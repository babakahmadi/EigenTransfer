
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
get_ipython().magic(u'matplotlib inline')


# <h2 style="color:pink"> load datasets</h2>

# In[2]:

glass = pd.read_csv('glass.data',header=None)
semeion = pd.DataFrame(np.loadtxt('semeion.data'))
wdbc = pd.read_csv('wdbc.data', header=None)
wpbc = pd.read_csv('wpbc.data', header=None)


# <h2 style="color:blue"> read glass dataset </h2>
# <ul>
#     <li>number of features: 9</li>
#     <li>number of classes: 7</li>
#     <li>number of data: 214</li>
# </ul>

# In[3]:

Xg = glass.drop([0,10],axis=1)
Yg = glass[10]
x1g = Xg[Yg==1]
x2g = Xg[Yg==2]
Xg.head()


# <h2 style="color:blue"> read semeion dataset </h2>
# <ul>
#     <li>number of features: 256, boolean</li>
#     <li>number of classes: 10</li>
#     <li>number of data: 1593</li>
# </ul>

# In[4]:

Ys = semeion[256]
for i in range(10):
    Ys[semeion[256+i]==1] = i+1
Xs = semeion.drop(range(256,266),axis=1)
Xs.shape


# <h2 style="color:blue"> read wdbc dataset </h2>
# <ul>
#     <li>number of features: 30</li>
#     <li>number of classes: 2</li>
#     <li>number of data: 569</li>
# </ul>

# In[5]:

Xwd = wdbc.drop([0,1],axis=1)
Ywd = (wdbc[1]=='M')
Xwd.shape


# <h2 style="color:blue"> read wpdc dataset </h2>
# <ul>
#     <li>number of features: 33</li>
#     <li>number of classes: 2</li>
#     <li>number of data: 198</li>
# </ul>

# In[6]:

Xwd = wpbc.drop([0,1],axis=1)
Ywd = (wpbc[1]=='M')
Xwd.head()


# <h2 style="color:purple"> PCA </h2>

# In[15]:

Components = 5
pca = PCA(n_components=Components)
transformedXg = pca.fit_transform(pd.concat([x1g,x2g]))

plt.plot(transformedXg[:x1g.shape[0],0],transformedXg[:x1g.shape[0],1],'o')
plt.plot(transformedXg[x1g.shape[0]:,0],transformedXg[x1g.shape[0]:,1],'x')
plt.show()


# <h2 style="color:purple"> LDA </h2>

# In[25]:

from sklearn.lda import LDA
lda = LDA(n_components=5)

xg = pd.concat([x1g,x2g])
yg = np.zeros(xg.shape[0])
yg[:x1g.shape[0]] = np.ones(x1g.shape[0])
transformedXg2 = lda.fit_transform(xg,yg)
print transformedXg2.shape
plt.plot(transformedXg2[:x1g.shape[0],0],np.zeros(x1g.shape[0]),'o')
plt.plot(transformedXg2[x1g.shape[0]:,0],np.zeros(x2g.shape[0]),'x')
plt.show()


# <h2 style="color:purple"> Supervised Graph</h2>

# In[ ]:




# <h2 style="color:purple"> Unsupervised Graph</h2>

# In[ ]:




# <h2 style="color:green"> Classify </h2>

# In[ ]:



