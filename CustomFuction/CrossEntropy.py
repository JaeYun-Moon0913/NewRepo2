
import torch 
import numpy as np 

class CompareFuc:
    def __init__(self, predict, label_seg):
        '''
        # input 
        predict : list , label_seg -> list
        predict = grad_Cam interpolation(224,224) pixel 
        label_seg = (224,224) segmentation GT(ground truth)
    
        # return 분산
        '''
         
        self.predict = predict
        self.label_seg = label_seg

        self.one_ch_lable = label_seg[:,:,0]
        self.one_ch_onehot =np.uint8(self.one_ch_lable/255)
        # # input grad_cam 
        # grad_cam(w*a).sum(1) relu , interpolate 
        # (grad_cam - grad_cam.min) / (max - min)  
        # type : torch, in cuda 
        predic_sq = self.predict.squeeze()
        self.ht_map  = predic_sq.cpu().numpy()
        self.ht_map_img = np.uint8(self.ht_map*255)
        self.pixel_onehot = np.uint8(self.label_seg / 255)


    def CrossEntropy_img(self):
        # x = 예측 
        # y = 정답
        delta = 1e-7 
        return -np.sum(self.one_ch_onehot*(np.log(self.ht_map_img+delta)/np.log(255))) 
    
    def CrossEntropy_np(self):
        # x = 예측 
        # y = 정답
        delta = 1e-7 
        return -np.sum(self.one_ch_onehot*np.log(self.ht_map+delta)) 


    def intersection(self):
        # one_ch_onehot = 0,1 로 이루어진 이미지 사진 
        result_np =np.multiply(self.one_ch_onehot,self.ht_map_img)
        # print('max result : ', result_np.max())
        # print('min result : ', result_np.min())

        print('shape : ', result_np.shape)
        # print('type : ',type(result_np))
        #print('result : ', result_np)

        result_np_sum = np.sum(result_np)

        
        return result_np_sum
