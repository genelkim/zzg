
# coding: utf-8

# In[49]:

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# In[44]:

data=[6100,6230]


# In[5]:

random.seed(123)


# In[16]:

mu,sigma=0,100


# In[26]:

np.random.normal(mu,sigma,1)[0]


# In[41]:

data.append(np.random.normal(mu,sigma,1)[0]+data[-1])


# In[42]:

data


# In[47]:

for i in range(3000):
    if data[-1]>=6300 or data[-1]<=5900:
        data.append(6100)
    else:data.append(np.random.normal(mu,sigma,1)[0]+data[-1])
    
    
    


# In[53]:

plt.figure(figsize=(15,15))
plt.plot(data ,'k-',lw=0.3)
plt.show()


# In[50]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
price = index_data['close']
price.plot(ax = ax, style = 'k-')
fig.show()

