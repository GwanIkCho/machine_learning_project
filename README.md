### 경마 승률예측

#### Features
         
 -  dtypes: bool(2), float64(381), int64(26), object(15)
 -  총 427개의 피쳐       

#### Target

 -  Win Odds : 승리 확률

    
---

- 데이터에 결측지, 중복행 X

- 먼저 전체 모델들을 이용하여 선형 회귀 예측을 수행
<img src='./image/img01.png'>

- 트리모델들의 R2가 매우 높은수치를 보이고 있어 과적합 가능성이 있고 확인필요

---
#### LGBM모델 사용

- Kfold cross_val_score를 통한 비교
<img src='./image/img02.png'>

-  Pytorch를 통해 loss를 통한 비교
<img src='./image/img03.png'>

-  LGBM모델 사용하였을때 학습데이터와 테스트 데이터의 대한 예측값
<img src='./image/img04.png'>

---

#### 차원축소

- LGBM모델을 이용 pca로 차원축소 진행
<img src='./image/img05.png'>

- 과적합 의심
<img src='./image/img06.png'>

- Ridge사용하여 과적합 방지
<img src='./image/img07.png'>
