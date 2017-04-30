
# coding: utf-8


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
random.seed(123)
#data=[6100,6230]
data=[20100,20230]
mu,sigma=-1,1
for i in range(3000):
    data.append(data[-1]-5)
    #+np.random.normal(mu,sigma,1)[0])
    #if data[-1]>=6300 or data[-1]<=5900:
    #    data.append(6100)
    #else:data.append(np.random.normal(mu,sigma,1)[0]+data[-1])
    
np.savetxt("data/linear_dummy_noise_-5_0.csv", data, delimiter=",")

