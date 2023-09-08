주제 : 열차의 주행 안정성 진단에 사용되는 ‘탈선계수’를 보다 높은 정확도로 예측할 수 있는 인공지능 모델 개발

기간 : 2023년 7월 17일 ~ 2023년 9월 18일

Top 7.3%(5/67)

### Score
평가 산식 : RMSE(Root Mean Squared Error)
private score : 3.28403
public score : 3.2741
데이터 전처리
고객의 나이가 100세 이상 또는 3세 이하일 경우 평균 나이로 대체

고객의 나이를 범주화(10대 ~ 60대 이상)

특수문자 제거

import re
train["Book-Author"] = train["Book-Author"].apply(lambda ba : re.sub(r'[^a-zA-Z\s]', '', ba))

# 좌우 공백 제거
train["Book-Author"] = train["Book-Author"].apply(lambda ba : ba.strip())
len(sorted(train["Book-Author"].unique())) # 89004

# 여러 개의 공백을 하나의 공백으로 변경
train["Book-Author"] = train["Book-Author"].apply(lambda ba : ' '.join(ba.split()))
len(sorted(train["Book-Author"].unique())) # 88649
단, 책 제목, 출판사의 경우 숫자로만 이루어진 경우가 있기 떄문에 특수문자만 제거(예 : 조지 오웰의 1984)
최대한 카테고리를 줄이기 위해 모든 문자열을 소문자로 변경, 공백 제거

고객의 거주 지역 null, 이상치 제거

단일 문자로만 이루어진 경우(예 : xx, aaaaa)
null값의 경우 na, n/a, "" 등 다양하게 표현되어 있음
문자열의 길이가 1인 경우
공백인 경우
Modeling
image

총 87만개의 데이터 중에서 평점이 0인 경우가 55만개, 1점과 2점은 각각 1만개 이하로 imbalance한 분포를 가지고 있음

oversampling, undersampling을 적용 했을 때 score가 약 3.9로 성능이 좋지 않았음
데이터가 imbalance하기 때문에 성능이 안좋게 나온 것으로 예상됨
weighted ensemble을 사용
AutoGluon(weighted ensemble) 사용

num_stack_levels = 2, 2 보다 클 경우 과적합 발생
피드백
출판년도 범주화
출판년도가 도서 평점과의 상관관계가 낮을 거라 생각해 범주화를 하지 않은 채로 모델링을 진행함
2등 코드를 확인해보니 1900년대 이후에는 10년 단위로 범주화하고 나머지 년도는 100년 또는 50년 단위로 범주화를 진행함
지역 구분 세분화
지역은 대부분 지역, 주, 국가로 나눠져 있어 지역, 주, 국가를 각각의 칼럼으로 나누려고 했으나 영국의 경우 5개의 구분으로 이루어져 있고 몇몇 국가들은 2개의 구분으로 나눠져 있었음
따라서 pycountry를 이용해 국가만 추출해 모델링을 진행했으나 성능이 가장 낮게 나왔음(Public Score :3.9)
이후에 구글링을 통해 나라별 주의 도시를 저장한 데이터가 있어서 사용하려 하였으나 외부 데이터를 사용하지 못하는 규정으로 인해 사용하지 못함
따라서 기존 지역 칼럼에서 간단한 전처리(특수문자, 결측치, 공백, 숫자 제거)만 적용한 데이터를 범주화함
연령대 구분
이상치에 해당하는 범위를 100 -> 90으로 변경, 3 -> 5로 변경
