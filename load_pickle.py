

import pickle
import numpy as np

prices = pickle.load(open('data/XBTEUR_1day.pkl', 'r'))
parr = np.array(prices)
np.savetxt('data/XBTEUR_1day.csv', parr, delimiter=',')

