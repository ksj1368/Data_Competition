# [DACON]대학 대항전: 퍼즐 이미지 AI 경진대회

## 대회 성과
- **최종 순위**: 33등 / 369팀 (상위 9%)
- **검증 점수**: 0.91973/0.92003(Public/Private)

## 대회 개요

- 4x4 퍼즐 이미지를 원래 이미지로 배열하는 AI 모델 개발

### 대회 링크
- [DACON 대회 페이지](https://dacon.io/competitions/official/236207/overview/description)

## Dataset

### 데이터 구조
- **INPUT**: 4x4로 퍼즐 이미지 (16개 조각)
- **OUTPUT**: 각 조각의 올바른 위치 (1-16 순서)
- **이미지 크기**: 256x256 픽셀
- **패치 크기**: 16x16 픽셀

### 데이터 전처리
- 퍼즐 조각들을 원래 위치로 복원 (`reset_image`)
- 학습 시 랜덤 셔플을 통한 데이터 증강 (`shuffle_image`)
- 인접 조각 간의 연결 관계 매트릭스 생성 (`get_adjacency_matrix`)

## 모델 아키텍처: JigsawElectra

### 전체 구조
JigsawElectra는 두 단계의 트랜스포머 아키텍처를 사용

#### 1단계: Local Connection Prediction
- **목적**: 각 퍼즐 조각 간의 연결 관계 예측
- **방법**: 트랜스포머를 사용하여 상하좌우 방향의 인접 패치 관계 파악
- **출력**: 연결 타입 (상: 1, 하: 2, 좌: 3, 우: 4, 연결없음: 0)

#### 2단계: Global Arrangement Prediction
- **목적**: 1단계 결과를 바탕으로 최종 퍼즐 배열 예측
- **특징**:
  - Piece-type embedding을 positional bias로 활용
  - Connect-type embedding을 attention bias로 활용
  - 16개 조각의 최적 재배열 순서 예측

### 주요 구성 요소

#### Backbone model
```python
model = timm.create_model('vit_medium_patch16_gap_256', pretrained=True, num_classes=0)
```

#### Module
- **Positional Embedding**: 16개 퍼즐 위치에 대한 학습 가능한 임베딩
- **Piece Type Embedding**: 10가지 퍼즐 조각 형태 (모서리, 코너, 중앙 등)
- **Connect Type Embedding**: 5가지 연결 관계를 attention head에 매핑

#### Loss Function
```python
loss = loss_local * 0.2 + loss_global
```
- Local loss: 연결 관계 예측 손실
- Global loss: 최종 배열 예측 손실

## Metric
### 스코어 계산
```python
score = (accuracy_1x1 + accuracy_2x2 + accuracy_3x3 + accuracy_4x4) / 4
```

- **1x1 정확도**: 개별 조각의 정확한 위치 비율
- **2x2 정확도**: 2x2 부분 퍼즐의 완전 일치 비율 (9개 영역)
- **3x3 정확도**: 3x3 부분 퍼즐의 완전 일치 비율 (4개 영역)  
- **4x4 정확도**: 전체 퍼즐의 완전 일치 비율

## Train
- **학습 시간**: 약 26분/epoch (Tesla T4(16GB) 1개)
### Model config
- **batch size**: 24
- **optimizer**: AdamW (lr=1e-5)
- **precision**: 16-bit mixed precision
- **seed**: 812

### data Split
- **train/valid ratio**: 80:20(random split)

### Callback setting
- **Checkpoint**: validation score 기준 상위 3개 모델 저장
- **Early stop**: 10 에포크 patience

## 핵심 혁신점

### 1. Two Stage Training
- 먼저 지역적 연결 관계를 학습
- 이를 바탕으로 전역적 배열을 예측

### 2. 다양한 임베딩 활용
- Piece-type embedding으로 퍼즐 조각 형태 인코딩
- Connect-type embedding으로 attention 메커니즘 강화

### 3. 멀티스케일 평가
- 다양한 크기의 부분 퍼즐 정확도를 종합적으로 평가


## Main Class

### JigsawDataset
퍼즐 이미지 데이터 로딩 및 전처리

```python
class JigsawDataset(Dataset):
    def __init__(self, df, data_path, mode='train')
    def reset_image(self, image, shuffle_order)      # 퍼즐 복원
    def shuffle_image(self, image)                   # 랜덤 셔플
    def get_adjacency_matrix(self, order)            # 연결 관계 매트릭스
    def get_score(self, order_true, order_pred)      # 평가 점수 계산
```

### JigsawElectra
메인 모델 아키텍처

```python
class JigsawElectra(nn.Module):
    def local_forward(self, x, label=None)           # 1단계: 연결 관계 예측
    def global_forward(self, x, piece_type, connect_type, label=None)  # 2단계: 전역 배열 예측
```

### LitJigsawElectra
PyTorch Lightning 래퍼 클래스

```python
class LitJigsawElectra(L.LightningModule):
    def training_step(self, batch)                   # 학습 스텝
    def validation_step(self, batch)                 # 검증 스텝
    def predict_step(self, batch)                    # 추론 스텝
```

