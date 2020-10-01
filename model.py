

import pandas as pd
import numpy as np
import pickle

x = np.arange(0,15).reshape(5,3)
y = np.array([7,15,15,23,11]).reshape(-1,1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)


pickle.dump(reg,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
c=model.predict(np.array([1,5,4]).reshape(-1,3))