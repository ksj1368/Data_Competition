### 주제 : 손글씨 인식 AI 모델 개발
- 기간 : 2022년 12월 26일 ~ 2023년 1월 16일
- 유아의 손글씨 인식 AI 모델 개발
- Top 10%(38/430)

## 데이터 전처리
- cutmix
  
  ![image](https://github.com/ksj1368/Data_Competition/assets/83360918/9865f3ff-7866-4551-b35e-717404876a89)
    - 문자열의 길이가 4, 5, 6인 데이터의 개수가 1, 2, 3에 비해 적기 때문에 cutmix 진행
  - 문자의 길이가 6인 데이터의 개수가 다른 데이터에 비해 현저히 적어 문자의 길이가 6이 되도록 2개 이상의 이미지를 합성
  ![image](https://github.com/ksj1368/Data_Competition/assets/83360918/cd40c661-cc0d-45a1-9831-1fb879785400)

- 블러링으로 데이터 증강

![image](https://github.com/ksj1368/Data_Competition/assets/83360918/81d15bb5-6a9c-44cd-97d1-bd74c173e299)

## Modeling
- CRNN(CNN + RNN)사용
  - CNN
    - 이미지에서 특징을 추출
  - RNN
    - 시퀀스 데이터를 처리하기 위해 사용
