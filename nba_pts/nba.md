### NBA 득점예측

#### Features
         
-  team_abbreviation: 팀 약어
-  age: 나이               
-  player_height: 선수 키     
-  player_weight: 선수 몸무게                              
-  draft_round: 드래프트 라운드       
-  draft_number: 드래프트 번호       
-  gp: 경기 수 (games played)                             
-  reb: 리바운드 수 (rebounds)                
-  ast: 어시스트 수 (assists)                
-  net_rating: 넷 레이팅 (팀이 코트에 있을 때의 순 득실점)         
-  oreb_pct: 공격 리바운드 비율 (offensive rebound percentage)           
-  dreb_pct: 수비 리바운드 비율 (defensive rebound percentage)           
-  usg_pct: 사용률 (usage percentage, 선수가 팀의 공격에서 차지하는 비율)            
-  ts_pct: 슈팅 성공률 (true shooting percentage, 효율적인 슈팅 성공률)             
-  ast_pct: 어시스트 비율 (assist percentage, 팀의 필드골 중 이 선수가 어시스트한 비율)            

#### Target

-  pts: 득점 (points)

---

- 데이터에 결측지, 중복행 X

- 먼저 전체 모델들을 이용하여 선형 회귀 예측을 수행
<img src='./image/nba01.png'>

- LinearRegression 의 R2가 0.9089로 매우 높은수치를 보이고 있어 과적합 가능성이 있고 확인필요



- Kfold cross_val_score를 통한 비교
<img src='./image/nba03.png'>

-  Pytorch를 통해 loss를 통한 비교
<img src='./image/nba02.png'>

-  LinearRegression 사용하였을때 학습데이터와 테스트 데이터의 대한 예측값
<img src='./image/nba04.png'>

---
#### 최적의 모델찾기

- Pipeline을 통해 StandardScaler 및 차원축소(2차원) 진행

- LGBMRegressor 모델을 통해 차원축소 적용

- Ridge 사용하여 규제적용(alpha=4000)

- Ridge 사용하여 규제적용(alpha=4000) 및 차원축소(2차원)

<img src='./image/nba05.png'>

