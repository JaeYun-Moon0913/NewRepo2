

import torch 
import numpy as np 

class CrossEntropy:
    def __init__(self, predict, label_seg):
    '''
    # input 
    predict : list , label_seg -> list
    predict = grad_Cam interpolation(224) pixel 
    label_seg = 224 segmentation GT(ground truth)
    
    # return 

    '''
    self.predict = predict
    self.label_seg = label_seg

    def ce(self.predict, self.label_seg):
        delta = 1e-7 
        return -np.sum(t*np.log(y+delta))
