import numpy as np

class State:
    def __init__(self, k=3):
        self.k = k
        self.shape = (k, 4)
        self.n = 4 * k 
    
    def reset(self):
        return np.random.rand(self.n)

         

