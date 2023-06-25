# level2_movierecommendation-recsys-09
level2_movierecommendation-recsys-09 created by GitHub Classroom

## 목차
### [Project Configuration](#project-configuration-1)
### [프로젝트 개요](#프로젝트-개요-1)
- [1. 프로젝트 주제 및 목표](#1-프로젝트-주제-및-목표)
- [2. 프로젝트 개발 환경](#2-프로젝트-개발-환경)
### [프로젝트 팀 구성 및 역할](#프로젝트-팀-구성-및-역할-1)
### [프로젝트 수행 내용 및 결과](#프로젝트-수행-내용-및-결과-1)
- [1. EDA](#1-eda)
- [2. 모델링](#2-모델링)
- [3. 성능 개선 및 앙상블](#3-성능-개선-및-앙상블)
- [4. 결과](#4-결과)
### [결론 및 개선 방안](#결론-및-개선-방안-1)

## Project Configuration
📦level2_movierecommendation-recsys-09  
 ┣ 📂base  
 ┣ 📂config  
 ┣ 📂data_loader  
 ┣ 📂experiment  
 ┣ 📂logger    
 ┣ 📂model  
 ┣ 📂test  
 ┣ 📂trainer  
 ┣ 📂utils  
 ┣ 📜.gitignore  
 ┣ 📜parse_config.py  
 ┣ 📜requirements.txt  
 ┣ 📜test.py  
 ┣ 📜train.py  
 ┗ 📜README.md

## 프로젝트 개요

### 1. 프로젝트 주제 및 목표

-	사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할, 선호할 영화를 예측  
- implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 시간 순서의 sequence에서 일부 item이 누락된 상황을 상정

### 2. 프로젝트 개발 환경

•	팀 구성: 5인 1팀, 인당 V100 서버를 VS Code와 SSH로 연결하여 사용  
•	협업 환경: Notion, GitHub, WandB


## 프로젝트 팀 구성 및 역할
-	😇 곽희원: EASE, Catboost 모델 구현 및 하이퍼파라미터 튜닝, 앙상블  
-	😄 김현정: EDA, 데이터 전처리, UltraGCN 모델 구현 및 하이퍼파라미터 튜닝  
-	😆 민현지: EDA, SASRec, BERT4Rec 모델 구현 및 하이퍼파라미터 튜닝, 앙상블  
-	🤭 이희만: EDA, Multi-VAE 모델 구현 및 하이퍼파라미터 튜닝, 앙상블  
-	😏 황찬웅: AutoRec 모델 구현 및 하이퍼파라미터 튜닝

  
## 프로젝트 수행 내용 및 결과

### 1. EDA

- 데이터 분석  
(1) train_rating.csv : user, item, time feature로 구성됨, 31360명의 고유한 user와 6807개의 고유한 item으로 구성  
(2) years.tsv : 6799개의 item에 대한 year 정보  
(3) writers.tsv : 5648개의 item에 대한 writer 정보가 존재, 2989명의 고유한 writer로 구성  
(4) directors.tsv :	5503개의 item에 대한 director 정보가 존재, 1340명의 고유한 director로 구성  
(5) genres.tsv : 18개의 고유한 genre로 구성  
(6) titles.tsv : 마지막 부분에 영화의 연도 정보 존재  

- 결측치 처리  
(1) year : 0.3%의 결측치 존재, title을 이용해서 결측치 처리  
(2) writer, director : 2.7%의 결측치 존재, 결측치를 채울 정보가 부족해 결측치를 others로 처리

- time 분석  
time을 기준으로 elapsed time을 확인한 결과 90% 이상이 10분 내에 발생한 interaction  
예전에 본 영화를 한 번에 평점을 매긴 데이터로 보여짐

### 2. 모델링  
(1) AutoRec : Recall@10 0.1163  
(2) BERT4Rec : Recall@10 0.0728  
(3) SASRec : Recall@10 0.0740  
(4) Catboost : Recall@10 0.1346  
(5) Multi-VAE : Recall@10 0.1339  
(6) EASE : Recall@10 0.1422  
(7) UltraGCN : Recall@10 0.1021  

### 3. 성능 개선 및 앙상블  
- WandB를 통한 모니터링, WandB sweep을 이용한 파라미터 튜닝  
- 앙상블  
![화면 캡처 2023-06-25 201737](https://github.com/boostcampaitech5/level2_movierecommendation-recsys-09/assets/91173904/1eb845d7-155b-436a-afd0-435ef16c4ba0)

### 4. 결과
EASE와 M-VAE를 앙상블 한 모델 채택  
Public 기준 0.1622, Private 기준 0.1625를 기록

## 결론 및 개선 방안
- 잘한 점  
  WandB를 이용해 하이퍼파라미터 튜닝을 하고 코드를 파이토치 템플릿에 적용해 일관성 있는 코드 작성 
  목표했던 timeline에 맞춰서 계획을 잘 지켰다
  
- 시도했으나 잘 되지 않은 점  
  Context aware 모델인 NFM 모델 구현을 시도했으나 모델링에 문제가 발생
  
- 아쉬웠던 점  
  사용자의 interaction 데이터만 사용한 모델이 주가 되며, 다른 feature들을 많이 사용하지 못함
  
- 프로젝트를 통해 배운 점  
  프로젝트 계획 수립의 중요성  
  협업과 커뮤니케이션을 통한 일관성 있는 결과 도출
