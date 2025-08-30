# 2025 전력사용량 예측 AI 경진대회

## 대회 개요

### 대회 정보
- **대회명**: 2025 전력사용량 예측 AI 경진대회
- **주최**: 한국에너지공단
- **주관**: DACON
- **대회 기간**: 2025.07.14 ~ 2025.08.25

### 대회 링크
- [대회 홈페이지](https://dacon.io/competitions/official/236531/overview/description)

### 대회 성적
- **참여 인원**: 1명
- **최종순위**: 934팀 중 42등 (상위 4.5%)
     <img width="879" height="92" alt="image" src="https://github.com/user-attachments/assets/c31da90d-6508-4b53-b80a-e9c465ce25df" />
- **Public Score**: 5.60161
- **Private Score**: 6.05345
- **평가 지표**: SMAPE (Symmetric Mean Absolute Percentage Error)


## 문제 정의

### 주제
- 건물의 전력사용량 예측 AI 모델 개발

### 설명
- 건물 정보와 기상 정보를 활용하여 특정 시점의 전력사용량을 예측하는 AI 모델을 개발합니다. 안정적이고 효율적인 에너지 공급을 위해서는 전력 사용량에 대한 정확한 예측이 필요

### 예측 대상
- 100개 건물의 2024년 08월 25일부터 2024년 08월 31일까지의 전력사용량(kWh)
- 시간 단위 예측 (7일 × 24시간 × 100개 건물 = 16,800개 예측값)

## 데이터셋 설명

### 데이터 구성
1. **train.csv**: 학습용 전력소비량 데이터
2. **building_info.csv**: 건물 개요 정보
3. **test.csv**: 테스트용 기상 데이터
4. **sample_submission.csv**: 제출 양식

### 데이터 세부 정보
- **건물 데이터**: 10개 유형, 100개 건물의 전력소비량 (1시간 주기, 2024.6.1~8.31)
- **기상 데이터**: 기온, 강수량, 풍속, 습도, 일조, 일사
- **건물 개요**: 연면적, 냉방면적, 태양광·ESS 용량 등
- **건물 유형**: 공공, 학교, 백화점, 병원, 아파트, 호텔 등

    ### 1. 훈련 데이터 (train.csv)
    - **기간**: 2024년 6월 1일 ~ 8월 31일 (3개월)
    - **건물 수**: 100개 건물
    - **시간 단위**: 1시간 간격
    - **타겟**: 전력소비량(kWh)
    - **기상 정보**: 기온(°C), 풍속(m/s), 습도(%), 강수량(mm), 일조(hr), 일사(MJ/m2)
    
    ### 2. 건물 정보 (building_info.csv)
    - **건물 특성**: 건물번호, 건물유형 (10가지: 공공, 학교, 백화점, 병원, 아파트, 호텔 등)
    - **면적 정보**: 연면적(m2), 냉방면적(m2)
    - **에너지 설비**: 태양광용량(kW), ESS저장용량(kWh), PCS용량(kW)
    
    ### 3. 테스트 데이터 (test.csv)
    - **예측 기간**: 2024년 8월 25일 ~ 31일 (1주일)

## Code
### 1. 데이터 전처리 (preprocess-2025-electricity-consumption.ipynb)

### 1. 데이터 로드 및 초기 처리
```python
# 성능 기여도가 낮은 피처 제거
train = train.drop(['일조(hr)', '일사(MJ/m2)'], axis=1)
```
- **목적**: 상관관계 분석 결과 중복성이 높은 피처 제거로 모델 복잡도 감소
- **근거**: 일조량과 일사량은 기온과 0.7 이상의 높은 상관관계

### 2. 건물 정보 범주화
```python
# 연면적 구간화
b_info['연면적구간'] = pd.cut(b_info['연면적(m2)'],
    bins=[0,15000,30000,45000,float('inf')],
    labels=['소형(<15k)','중형(15k-30k)','대형(30k-45k)','초대형(45k+)'])
```
- **연면적 구간화**: 4단계 (소형/중형/대형/초대형)
- **냉방면적 구간화**: 4단계 구간 분할
- **목적**: 연속형 변수를 범주형으로 변환하여 Tree 기반 모델의 분할 성능 향상

### 3. 시간대 피처 생성
#### 3.1 기본 시간 피처
```python
df['month'] = df['일시'].dt.month
df['week_of_year'] = df['일시'].dt.isocalendar().week
df['dow'] = df['일시'].dt.dayofweek  # 요일 (0=월요일)
df['hour'] = df['일시'].dt.hour
```

#### 3.2 휴일/공휴일/건물 휴무무일 처리
- **일반 휴일**: 토/일요일, 공휴일 (현충일, 광복절)
- **건물별 휴무일**: 건물 유형에 따른 개별 휴무 패턴
  - 일요일만 휴무: [18번 건물]
  - 토/일 휴무: [2,3,5,6,8,12... 총 38개 건물]
  - 개별 특수 휴무일: 건물별 상세 휴무 일정 (7일차~94일차)

#### 3.3 순환 인코딩(Cyclical Encoding)
```python
# 시간의 주기성을 sin/cos로 표현
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
```
- **목적**: 시간의 순환성 (23시 다음이 0시) 모델이 학습할 수 있도록 변환
- **결과**: Linear 모델에서도 시간 경계의 연속성 학습 가능

#### 3.4 업무시간 및 전력 소요가 많은 시간대 플래그 피처
```python
df['business_hours'] = ((df['hour'].between(9, 18)) & 
                       (df['dow'] < 5) & 
                       (df['is_building_holiday'] == 0)).astype(int)
df['peak_flag'] = df['hour'].between(12, 16).astype(int)
```

### 4. 기상 피처 생성성

#### 4.1 열지수(Heat Index)
```python
df['heat_index'] = (
    -8.784695 + 1.61139411 * T + 2.338549 * RH
    - 0.14611605 * T * RH - 0.012308094 * T**2
    - 0.016424828 * RH**2 + 0.002211732 * T**2 * RH
    + 0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2)
```
- **목적**: 기온과 습도의 복합적 영향을 하나의 지표로 표현
- **활용**: 체감온도가 전력소비(냉방)에 미치는 영향 정량화

#### 4.2 냉방도일(Cooling Degree Days)
```python
base = 24  # 기준온도
df['CDD'] = (T - base).clip(lower=0)
```
- **개념**: 기준온도(24°C) 이상일 때의 온도 누적
- **의미**: 건물 에너지 관리에서 사용되는 표준 냉방 수요 지표

#### 4.3 불쾌지수 및 체감온도
```python
# 불쾌지수
df['discomfort_index'] = 9/5*T - 0.55 * (1 - RH/100.0) * (9/5*T - 26) + 32

# 체감온도 (한국산업안전보건공단 공식)
df['feels_like_temp'], df['wet_bulb'] = calculate_feels_like_temperature(T, RH)
```

#### 4.4 폭염 및 열대야 플래그
```python
df['heat_wave_flag'] = (T >= 33).astype(int)  # 폭염
df['tropical_night_flag'] = (((df['hour'] >= 18) | (df['hour'] <= 9)) & (T >= 25)).astype(int)  # 열대야
```

### 5. 시계열 피처 생성 (Data Leakage 방지)

#### 5.1 Data Leakage 방지 전략
```python
def add_advanced_time_series_features_separate(train_df, test_df):
    """Train과 Test를 완전 분리하여 시계열 피처 생성"""
```
- **핵심**: Train과 Test 데이터를 완전히 분리하여 처리
- **목적**: 미래 정보가 과거 예측에 사용되는 것을 방지

#### 5.2 차분(Differencing) 피처
```python
df[f'{col_name}_diff1'] = grouped.diff(1)    # 1시간 전 대비 변화량
df[f'{col_name}_diff3'] = grouped.diff(3)    # 3시간 전 대비 변화량  
df[f'{col_name}_diff24'] = grouped.diff(24)  # 24시간 전(전일) 대비 변화량
```
- **목적**: 기상 변수의 변화율을 통한 전력소비 패턴 예측

#### 5.3 지연(Lag) 피처
```python
for lag in [1, 3, 24]:
    df[f'{col_name}_lag{lag}'] = grouped.shift(lag)
```
- **의미**: 과거 기상 정보가 현재 전력소비에 미치는 지연 효과

#### 5.4 롤링 통계 피처
```python
for w in [3, 24]:  # 3시간, 24시간 윈도우
    roll = shifted.rolling(window=w, min_periods=w)
    df[f'{col_name}_mean{w}'] = roll.mean()
    df[f'{col_name}_median{w}'] = roll.median()
    df[f'{col_name}_range{w}'] = roll.max() - roll.min()
    df[f'{col_name}_median_mean_diff{w}'] = roll.median() - roll.mean()
```

### 6. 건물별 통계 피처 (Target Encoding)

#### 6.1 Data Leakage 방지 통계 계산
```python
# Train 데이터에서만 통계 계산
dow_hour_mean = train.groupby(['건물번호', 'dow', 'hour'])[target_col].mean()
hour_mean = train.groupby(['건물번호', 'hour'])[target_col].mean()
hour_std = train.groupby(['건물번호', 'hour'])[target_col].std()
```

#### 6.2 적용 전략
- **Train 적용**: 계산된 통계를 그대로 적용
- **Test 적용**: 누락된 조합은 전체 평균으로 대체
- **목적**: 건물별 시간대별 전력 사용 패턴 학습

### 7. 건물 클러스터링

#### 7.1 클러스터링 기법
```python
# 시계열 패턴 + 건물 특성 결합
ts_scaled = StandardScaler().fit_transform(pivot_filtered)  # 시계열 데이터
st_scaled = StandardScaler().fit_transform(stats_filtered)   # 건물 특성
X_cluster = np.hstack([ts_scaled, st_scaled])

# 계층적 클러스터링
Z = linkage(X_cluster, method='ward')
cluster_labels = fcluster(Z, t=5, criterion='maxclust')
```

#### 7.2 클러스터링 요소
- **시계열 패턴**: 건물별 일별 평균 전력소비 패턴
- **건물 특성**: 연면적, 냉방면적, 건물유형, 태양광용량 등
- **결과**: 5개 클러스터로 건물 분류

### 8. 최종 데이터 처리

#### 8.1 라벨 인코딩
```python
cat_cols = ['건물번호', '건물유형', 'is_holiday', 'is_building_holiday', 
           'is_weekday', 'time_period', 'peak_flag', '연면적구간', 
           '냉방면적구간', 'rain_flag', 'discomfort_category', 
           'heat_wave_flag', 'tropical_night_flag']
```

#### 8.2 결측치 처리
```python
# 태양광/ESS 관련 결측치를 0으로 처리
df['태양광용량(kW)'] = df['태양광용량(kW)'].replace('-', 0.0).astype(float)
# 나머지는 전진/후진 채우기
df[numeric_cols] = df[numeric_cols].bfill().ffill()
```

#### 8.3 이상치 처리
- 전력소모가 급격하게 감소/증가하는 전력소모량 제거
```python
out_date_dict = {
    5: ['5_20240804 07', '5_20240804 08'],
    8: ['8_20240721 08'],
    12: ['12_20240721 08', '12_20240721 09', '12_20240721 10', '12_20240721 11'],
    30: ['30_20240713 20', '30_20240725 00'],
    40: ['40_20240714 00'],
    41: ['41_20240622 01', '41_20240622 04', '41_20240717 14', '41_20240717 15'],
    42: ['42_20240717 14'],
    43: ['43_20240610 17', '43_20240610 18', '43_20240812 16', '43_20240812 17'],
    44: ['44_20240630 00', '44_20240630 02', '44_20240606 13', '44_20240606 14'],
    52: ['52_20240810 00', '52_20240810 02'],
    53: ['53_20240615 08', '53_20240615 11'],
    67: ['67_20240610 17', '67_20240610 18', '67_20240812 16', '67_20240812 17'],
    68: ['68_20240628 23', '68_20240629 01'],
    70: ['70_20240605 09', '70_20240603 11', '70_20240603 12'],
    72: ['72_20240721 11'],
    76: ['76_20240603 13', '76_20240620 12', '76_20240620 16'],
    79: ['79_20240819 04', '79_20240819 03', '79_20240819 05'],
    80: ['80_20240720 10', '80_20240720 11', '80_20240720 12', '80_20240706 10', '80_20240706 13', '80_20240706 14'],
    81: ['81_20240717 14'],
    90: ['90_20240605 18'],
    92: ['92_20240717 18', '92_20240717 19', '92_20240717 21'],
    94: ['94_20240727 09', '94_20240727 12'],
    97: ['97_20240605 05'],
}
```

#### 9. 피처 선택
- Feature Importance와 후진제거법을 사용하여 CV SMAPE가 감소하는 피처를 선택
```python
exclude_cols = ['일시', 'num_date_time', 'is_weekday', 'time_period', 'business_hours', 'rain_flag', 'max_days',
                'high_temp_flag', 'wind_chill_effect', 'temp_change_hour', 'temp_change_day', '기온_C_diff1', 
                '기온_C_diff3', '기온_C_diff24', '기온_C_lag1', '기온_C_lag3', '기온_C_lag24', '풍속_ms_diff1', 
                '풍속_ms_diff3', '풍속_ms_diff24', '풍속_ms_lag1', '풍속_ms_lag3', '풍속_ms_lag24', '습도_pct_diff1',
                '습도_pct_diff3', '습도_pct_diff24', '습도_pct_lag1', '습도_pct_lag3', '습도_pct_lag24', '기온_C_mean3', 
                '기온_C_median3', '기온_C_range3', '기온_C_median_mean_diff3', '기온_C_mean24', '기온_C_median24', '기온_C_range24', 
                '기온_C_median_mean_diff24', '풍속_ms_mean3', '풍속_ms_median3', '풍속_ms_range3', '풍속_ms_median_mean_diff3', 
                '풍속_ms_mean24', '풍속_ms_median24', '풍속_ms_range24', '풍속_ms_median_mean_diff24', '습도_pct_mean3', '습도_pct_median3', 
                '습도_pct_range3', '습도_pct_median_mean_diff3', '습도_pct_mean24', '습도_pct_median24', '습도_pct_range24', 
                '습도_pct_median_mean_diff24']
```
- 최종 선택 피처
  ```python
  # Number of features: 43
    selected_features =  ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형', '연면적(m2)', 
                          '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적구간', 
                          '냉방면적구간', 'month', 'week_of_year', 'dow', 'day', 'hour', 'is_holiday', 
                          'is_building_holiday', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin',
                          'month_cos', 'dow_sin', 'dow_cos', 'peak_flag', 'cooling_ratio', 'temp_cool_ratio', 
                          'heat_index', 'CDD', 'discomfort_index', 'discomfort_category', 'heat_wave_flag', 
                          'tropical_night_flag', 'feels_like_temp', 'wet_bulb', 'dow_hour_mean',  'hour_mean',
                          'hour_std', 'cluster_id']
  ```

## model 
- XGBoost, LGBM, Crossformer로 실험 진행
### 실험 결과
- 단일 XGBoost 앙상블의 예측 성능이 가장 뛰어나 **단일 XGBoost** 앙상블만 활용
| 모델 | 구성/가중치 | SMAPE |
|--|--|--|
| xgb best1 + best2 | 0.8 / 0.2 | 5.6016190137 |
| LGBM | - | 6.1574841025 |
| Crossformer | - | 8.6690988677 |

## etc
### 1. 과소추정에 더 큰 패널티 부여
```python
def weighted_mse(alpha=3):
    # 과소추정에 3배 가중치 적용
    grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
```

### 2. 교차검증 전략
- **건물유형별 분할**: 일반화 성능 확보
- **분할 기준**: 마지막 주차(2024-08-18 ~ ) 데이터를 검증용으로 사용

### 3. 하이퍼파라미터 튜닝
- Optuna 사용

## 성능 개선 포인트
- 커스텀 손실함수 적용
- 도메인 피처 활용
- LGBM, CrossFormer를 사용해봤지만 예측 성능이 좋지 않아 제외
