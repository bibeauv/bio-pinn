from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Normalization:
    
    def __init__(self):
        pass
    
    def minmax(self, x):
        x = np.reshape(x, (-1,1))
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        return x, scaler