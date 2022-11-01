
import os
import cv2
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from Guided_Backprop.Guided_Backprop import GuidedBackpropReLUModel, deprocess_image

import numpy as np #numpy library
#np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 


class mv:
    def __init__(self,layer,model,img,origin_img,label_idx ,name,save_pth,label_list,re=False):
        self.f_conv = layer 
        self.orig_img = origin_img
        self.model = model
        self.f_name = name 
        self.pth = save_pth 
        self.img = img 
        self.activations = []
        self.gradients= []
        self.label_idx = label_idx
        self.label_list = label_list
        self.re = re


    # 마지막 layer 에서 activation, gradient 저장 
    def hook_feature(self,module, input, output):
        self.activations.append(output.cpu().data)
    def backward_hook(self,module, input, output):
        self.gradients.append(output[0])


    def make_img(self):
        self.model.eval()
        self.model._modules.get(self.f_conv).register_forward_hook(self.hook_feature)
        self.model._modules.get(self.f_conv).register_backward_hook(self.backward_hook)
        # guided_

        
        logit = self.model(self.img)
        score = logit[:,self.label_idx].squeeze()
        select_cls = self.label_list[self.label_idx]
        
        #print(score, self.f_name)
        select_sco = float(logit[:,self.label_idx].squeeze().cpu().data.numpy())
        select_sco = str(round(select_sco,3))

        predic_idx = np.argmax(logit.cpu().data.numpy())

        predic_sco = float(logit[:,predic_idx].squeeze().cpu().data.numpy())
        predic_sco = str(round(predic_sco,3))

        predic_cls = self.label_list[predic_idx]
        score.backward(retain_graph = True)

        act = self.activations[0].cuda() # (1, 512, 7, 7), forward activations

        grad = self.gradients[0] # (1, 512, 7, 7), backward gradients
        b, k, u, v = grad.size()

        alpha = grad.view(b, k, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
        wet = alpha.view(b, k, 1, 1) # (1, 512, 1, 1)
        
        grad_cam_map = (wet*act).sum(1,keepdim = True)
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = F.interpolate(grad_cam_map,size = (224,224),mode = 'bilinear',align_corners = False)
        map_min , map_max = grad_cam_map.min(), grad_cam_map.max() 
        grad_cam_map = (grad_cam_map - map_min).div(map_max-map_min).data
        self.grad_cam_mp = grad_cam_map

        grad_ht = cv2.applyColorMap(np.uint8(255*grad_cam_map.squeeze().cpu()),cv2.COLORMAP_JET)
        if self.re: 
            cv2.imwrite(os.path.join(self.pth +'/'+self.f_name+'_'+ predic_sco+'_'+predic_cls+'.jpg'),grad_ht)

        
        
        else: 
            # grad_cam_map 값 뽑기 
            txt_par_pth =os.path.join(os.getcwd(),'./txt_result_grad_cam')
            text_pth = os.path.join(txt_par_pth, self.f_name+'_'+select_sco+'_'+select_cls+'.txt')
            #print(grad_cam_map.shape)
            f = open(text_pth,'w')
            f.write(str(grad_cam_map.squeeze().cpu().numpy()))
            f.close()
            
        cv2.imwrite(os.path.join(self.pth +'/'+ self.f_name+'_'+ select_sco+'_'+select_cls+'.jpg'),grad_ht)


        # grad_cam + origin_img
        grad_ht  = np.float32(grad_ht) / 255 
        img = self.orig_img.squeeze().cpu().detach().numpy() 
        img = np.transpose(img,(1,2,0))

        grad_result = grad_ht +img 
        grad_result = grad_result / np.max(grad_result)
        grad_result = np.uint8(255 * grad_result)

        cv2.imwrite(os.path.join(self.pth + '/'+self.f_name+ '_'+select_sco+'_'+predic_cls +'_result'+ '.jpg'),grad_result)

        # if predic_cls != self.label_idx:
        #     score_dif = logit[:,predic_cls].squeeze()
        #     dif_score = float((logit[:,predic_cls].squeeze().cpu().data.numpy()))
        #     dif_score = str(dif_score)
        #     score_dif.backward(retain_graph = True)
        #     dact = self.activations[0].cuda() # (1, 512, 7, 7), forward activations

        #     dgrad = self.gradients[0] # (1, 512, 7, 7), backward gradients
        #     db, dk, u, v = dgrad.size()

        #     dalpha = dgrad.view(db, dk, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
        #     dwet = dalpha.view(db, dk, 1, 1) # (1, 512, 1, 1)
            
        #     dgrad_cam_map = (dwet*dact).sum(1,keepdim = True)
        #     dgrad_cam_map = F.relu(dgrad_cam_map)
        #     dgrad_cam_map = F.interpolate(dgrad_cam_map,size = (224,224),mode = 'bilinear',align_corners = False)
        #     dmap_min , dmap_max = dgrad_cam_map.min(), dgrad_cam_map.max() 
        #     dgrad_cam_map = (dgrad_cam_map - dmap_min).div(dmap_max-dmap_min).data

        #     dgrad_ht = cv2.applyColorMap(np.uint8(255*dgrad_cam_map.squeeze().cpu()),cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join(self.pth +'/'+ '+++' +dif_score+'_'+predic_cls+'_'+self.f_name),dgrad_ht)

        return grad_cam_map, predic_idx, predic_cls

    def grad_cam_return(self):
        return self.grad_cam_mp

# 모델을 계속해서 불러와야 되서 CUDA out of memory 
class Guided_Grad_CAM():
    def __init__(self, model, origin_img, label_idx, grad_htmap,f_name, pth):
        self.models = model
        self.origins_img = origin_img
        self.label_idxs = label_idx 
        self.grad_cam_map = grad_htmap
        self.pth = pth 
        self.f_name = f_name
        gb_model = GuidedBackpropReLUModel(model = self.models,use_cuda = True)
        gb_num = gb_model(self.origins_img,target_category = self.label_idxs)
        #gb = deprocess_image(gb_num)
        # gb_model = GuidedBackpropReLUModel(model = self.model,use_cuda = True)
        # gb_num = gb_model(self.orig_img,target_category = self.label_idx)
        # gb = deprocess_image(gb_num)

        grad_cam_htmap = self.grad_cam_map
        grayscale_cam = grad_cam_htmap.squeeze(0).cpu().numpy() # (1, 224, 224), numpy
        grayscale_cam = grayscale_cam[0, :] # (224, 224)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb_num)

        cv2.imwrite(os.path.join(self.pth + 'Guided_Grad_CAM_' + self.f_name), cam_gb)


        
        