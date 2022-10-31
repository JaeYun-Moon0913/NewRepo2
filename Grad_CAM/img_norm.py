import os 
import glob 

import torch 
import torch.nn.functional as F

def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)   

class transform_img:
    def __init__(self,img_pth,means,stds):
        self.img_pth = img_pth
        self.mean = means
        self.std = stds
    def __call__(self):
        return self.trans_img
        
    def trans_img(self): # img_pth 주면 img -> torch tensor type (1,3,224,224) 
        # means = [r,g,b] / std = [r,g,b]
        p_img = Image.open(self.img_pth) 
        tensor_img = torch.from_numpy(np.asarray(p_img)).permute(2,0,1).unsqueeze(0).float().div(255).cuda()
        # vgg -> 1,3,224,224
        # alex -> 1,3,227,227
        if tensor_img != (1,3,224,224) :
            tensor_img = F.interpolate(tensor_img,size=(224,224),mode='bilinear', align_corners=False)

        norm_mean_std = Normalize(mean=self.mean, std=self.std)
        nt_img = norm_mean_std(tensor_img)

        return nt_img


"""
1. 이미지 하나 가져와서 pillow를 활용해서 이미지 열기 
    - Image.open('이미지 주소')
    ```
    from PIL import Image
    pil_img = Image.open(img_path)
    ```
2. 이미지 픽셀 정규화 해주기 위해 
    - pillow type -> numpy 로 바꿔주고, (224,224,3) -> (3,224,224)로 바꿔 준다.  
    - 맨 앞차원 늘려준다. (1,3,224,224)
    - 각 픽셀을 float형으로 바꿔주며, /255 해준다. 
    - 이렇게 만든 numpy를 torch tensor 형으로 바꿔주고 gpu에 넣어준다. 
    
    ```
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(DEVICE)
    ```
    
3. 이미지 사이즈 바꿔준다. (이미 opencv로 바꿔 줌 - 보간법(interpolate) 사용 )
    ```
    torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False) # (1, 3, 224, 224)
    ```
    - align_cornets : 아직 모르겠음 
    - mode : 보간법 방법 입력 하는 부분 종류는 아직 모름
    """