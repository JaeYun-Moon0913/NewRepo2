# 폴더 Grad_CAM 
## img_norm.Normalize 
### input : mean, std 
### return (1,3,n,n) - 각 픽셀 값을 mean값으로 빼고 std 값으로 나눈 픽셀에 대한 이미지 

- 그 외 transform_img class는 iterate로 뽑기가 어려움(return이 class type으로 나옴 )
- for문을 돌려서 결과물이 나오지 않아 사용 안함.. 
- 이미지 각각에 nomalize 하기 위한 클래스 


## model_value.mv 
### input : layer, model , img , origin_img,label_idx ,file_name, save_pth 
### return : 이미지 저장(grad_cam heatmap, grad_cam hmap + origin img), score 점수 


# VGG 
- xai : score가 낮은게 더 GT 에 더 많이 포함된다. 

## Unreal에서 구한 이미지 - grad_cam 한번 뽖아보기 
## 내일 만나는 분 대비해서 자료 작성하기 - 노션 작성 중 
## resnet 도 구현하기 
