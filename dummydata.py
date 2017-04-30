
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
random.seed(123)

#sma  simple moving average
def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)


mu,sigma=10,2
data=[6100,6230]
for i in range(50):
    data.append(data[-1]+np.random.normal(mu,sigma,1)[0])
    
data=np.array(data)    
dif=np.diff(data)
#dif means the difference of price between today and yesterday
dif=np.insert(dif,0,0)
#sma is the simple moving average of 15 days
sma15=calcSma(data,15)
d={'price':data,'dif':dif,'sma15':sma15}
d=pd.DataFrame(d)
d.fillna(0)
d.to_csv('data/data_10__dif.csv',index=False)
