# git 

git init 

git config user.name # 확인 없으면 추가 
git config user.email # 확인 없으면 추가 

git add .
git commit -m '~' 

git remote -v  # 리모트 저장소 이름 확인 및 https 
git remote add '리모트 저장소 이름' '주소-HTTPS'

git push origin master 
git push origin +master  # 변경할게 많으면 오류가 발생하는데 
# 이럴때 무시하고 올릴수 있다. 

https://git-scm.com/book/ko/v2/Git%EC%9D%98-%EA%B8%B0%EC%B4%88-%EB%A6%AC%EB%AA%A8%ED%8A%B8-%EC%A0%80%EC%9E%A5%EC%86%8C