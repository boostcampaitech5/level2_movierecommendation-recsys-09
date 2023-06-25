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
- [2. Feature Engineering](#2-feature-engineering)
- [3. 모델링](#3-모델링)
- [4. 성능 개선 및 앙상블](#4-성능-개선-및-앙상블)
- [5. 결과](#5-결과)
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
-	🤭 민현지: EDA, SASRec, BERT4Rec 모델 구현 및 하이퍼파라미터 튜닝, 앙상블  
-	🤗 이희만: EDA, Multi-VAE 모델 구현 및 하이퍼파라미터 튜닝, 앙상블  
-	😏 황찬웅: AutoRec 모델 구현 및 하이퍼파라미터 튜닝

  
