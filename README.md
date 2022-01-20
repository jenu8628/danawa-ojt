# TIL

### 폴더 경로

- docker
  - compose
    - test

- GitFlow
- 머신러닝
  - DACON
    - 와인품질분류
    - 타이타닉생존자예측
  - TensorFlow
  - 데이터분석(1주차)
    - 1_기온공공데이터
    - 2_데이터시각화기초
    - 3_인구공공데이터
    - 4_데이터분석라이브러리프로젝트
  - 부스트코스-템서플로우로시작하는딥러닝
    - 1.Basic_ML
  - 혼자공부하는머신러닝+딥러닝





# 1주차

## 2022/01/03(월)

- 프로그램 환경 세팅
  - Sametime, RocketChat, VS Code

## 2022/01/04(화)

- 필요 프로그램 및 IDE 설치
  - 도커, git, 소스트리, 파이참, Python(v-3.8)
- CSV파일에서 데이터 읽어오기
- matplotlib 라이브러리 사용법 익히기
- plot()함수를 이용한 기본 그래프 그리기
- 그래프 옵션 추가하기
- 기온 변화를 그래프로 나타내기
- 참고자료 : 모두의 데이터 분석 with 파이썬

## 2022/01/05(수)

- 히스토그램
- 상자그림
- 다양한 형태로 인구 구조 시각화
  - 기본 그래프
  - 막대 그래프, 항아리 모양 그래프
  - 파이 차트
  - 버블 차트, 산점도
- 대중교통 데이터 프로젝트
  - 막대 그래프
  - 꺾은선 그래프
- 참고자료 : 모두의 데이터 분석 with 파이썬

## 2022/01/06(목)

- matplotlib 라이브러리의 pyplot 모듈 명령어 정리
- numpy 라이브러리 기초 학습 및 인구 구조 분석 실습
- pandas 라이브러리 기초 학습 및 인구 구조 분석 실습
- 참고자료 : 모두의 데이터 분석 with 파이썬

## 2022/01/07(금)

- git flow 학습 및 실습
  - 파이참 git Flow Integration을 이용한 Git 브랜치 관리
- 텐서플로우 튜토리얼
- 머신러닝 개념 학습 및 경사하강 알고리즘 코드 구현
  - 선형회귀(Linear Regression), 가설(Hypothesis), 비용함수(Cost function), 경사하강법(Gradient Descent Algorithm), 볼록함수(Convex function) 학습

# 2주차

## 2022/01/10

- Docker 
  - 기본 개념 학습
    - 이미지, 컨테이너, 볼륨
  - 이미지 명령어 정리, 실습
  - 컨테이너 명령어 정리 및 실습
  - 볼륨 명령어 정리 및 실습
- JIRA 사용법
  - 포트 요청 이슈 생성
- Wike 사용법
  - 홈 작성
  - 통합 View와 홈 연결

## 2022/01/11

- 사내 VPN 설정
- Docker
  - Dockerfile 학습
  - Dockerfile 작성
    - ubuntu를 베이스로 python 실행
    - code-server 실행

## 2022/01/12

- Docker
  - Docker-compose 학습
    - 야믈코드 작성법
  - docker-compose.yml 작성
    - 기존에 작성한 Dockerfile과 mysql이미지를 이용한 야믈코드 작성
  - Docker Swarm 학습

## 2022/01/13

- Docker
  - Docker swarm
    - virtualbox에서 docker swarm 실습
    - AWS 위에서 도커 스웜 클라스터 적용
- AWS
  - AWS 를 활용한 컨테이너 서비스 개념 학습
  - EC2, S3, IAM, VPN, Route53, CloudWatch, Lambda, ElasticCache

## 2022/01/14

- 머신러닝
  
  - 선형회귀 학습 및 실습 (+ 미분)
  
  - 로지스틱 회귀 학습 및 실습
  
  - 다중 입력에 대한 실습
  
  - 벡터와 행렬 연산 학습
  
  - 소프트맥스 회귀 학습 및 실습

# 3주차

## 2022/01/17

- 머신러닝
  
  - 결정 트리
    
    - 불순도
    
    - 가지치기
    
    - 특성 중요도
  
  - 검증
    
    - 검증 세트
    - 교차 검증
  
  - 하이퍼파라미터 튜닝
    
    - 그리드 서치
    - 랜덤 서치
  
  - 앙상블 학습
    
    - 랜덤 포레스트
    - 엑스트라 트리
    - 그레디언트 부스팅

## 2022/01/18

- 머신러닝
  
  - [DACON 와인 품질 분류](https://dacon.io/competitions/open/235610/overview/description) 
  - EDA(탐색적 데이터 분석)
  
    - matplot 사용
  - 여러가지 지도학습 알고리즘 이용하기
    - 히스토그램 기반 그래디언트 부스팅, 랜덤 포레스트,엑스트라 트리

## 2022/01/19

- [DACON 와인 품질 분류](https://dacon.io/competitions/open/235610/overview/description) 
  - class 및 함수를 이용한 코드 추상화

- [타이타닉 생존자 예측](https://dacon.io/competitions/open/235539/overview/description)
  - EDA(탐색적 데이터 분석)
    - matplot 사용

## 2022/01/20

- [타이타닉 생존자 예측](https://dacon.io/competitions/open/235539/overview/description)
  - 다양한 앙상블학습 이용하기
    - 히스토그램 기반 그래디언트 부스팅, LightGBM
    - 과대적합을 줄이고 예측율을 높이기 위한 데이터 분석 재시도
      - 결측치 처리에 관한 문제 해결
