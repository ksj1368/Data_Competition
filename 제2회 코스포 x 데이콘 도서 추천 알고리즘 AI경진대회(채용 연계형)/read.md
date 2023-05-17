### 주제 : 도서 추천 알고리즘 AI 모델 개발

- 기간 : 2023년 4월 17일 ~ 2023년 5월 15일
- Top 2%(11/679)
- private score : 3.28403
- public score : 3.2741

## 데이터 전처리
- 고객의 나이가 100세 이상 또는 3세 이하일 경우 평균 나이로 대체
- 고객의 나이를 범주화(10대 ~ 60대 이상)
- 특수문자, 숫자 제거
  - 단, 책 제목, 출판사의 경우 숫자로만 이루어진 경우가 있기 떄문에 특수문자만 제거(예 : 조지 오웰의 1984)

- 고객의 거주 지역 null, 이상치 제거
  - 단일 문자로만 이루어진 경우(예 : xx, aaaaa, slslslsl)
  - null값의 경우 na, null, Nan 등 다양하게 표현되어 있음
  - 문자열의 길이가 1인 경우
  - 공백인 경우

## Modeling
- 총 87만개의 데이터 중에서 평점이 0인 경우가 55만개, 1점과 2점은 각각 1만개 이하로 imbalance한 분포를 가지고 있음
  -  oversampling, undersampling을 진행하 봤을 때 score가 낮게 나와서 weighted ensemble을 사용
  -  
- AutoGluon(weighted ensemble) 사용

