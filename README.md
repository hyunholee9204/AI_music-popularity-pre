# AI_music-popularity-pre

## 프로젝트명: 🎵AI기반 노래 인기도 예측 모델🎵
- AI 기반 오디오 분석을 통해 AI가 5초만에 만들어낸 노래의 인기도를 예측하는 모델을 구축합니다.

### 프로젝트 목적
- 최근 인공지능 기술의 굉장한 발달로 음악 생성 또한 AI의 영역으로 확장되고 있습니다.<br> 본 프로젝트는 AI가 생성한 음악이 실제로 어느 정도의 완성도와 대중성을 갖추고 있는지, 사람들에게 인기를<br> 끌 수 있는 요소를 얼마나 반영하고 있는지를 탐색하는 것을 목적으로 합니다. AI 음악의 인기도를<br> 예측하는 모델을 구축하여, 그 가능성과 활용 가치를 분석하고자 합니다.
  
### 데이터셋
- Kaggle에서 제공하는 스포티파이 인기 차트 데이터셋으로, 전 세계 70개국 이상에서 인기 있는 상위 50곡의 정보를 수집한 음악 트렌드 데이터입니다.<br>
데이터셋 링크: https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated

### AI가 생성한 노래 5곡, 모델 파일 링크
https://1drv.ms/u/c/f235d97931f0c634/Ee1Yd8335blOnsBqwH-AsNEBEaRHhhZfQDHx5Xphnwaqdg?e=PN9kOy

### 사용 기술
- python, pandas, scikit-learn, librosa, matplotlib 등

spotify_trend_premodel.py(모델 학습)에서 사용한 라이브러리
```python
import pandas as pd  # 표 형태(데이터프레임)의 데이터를 다룰 수 있게 해줌
import numpy as np  # 수학 계산(배열, 행렬 계산 등)을 빠르게 처리
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 나눔
from sklearn.ensemble import RandomForestRegressor  # 인기도를 예측하는 랜덤 포레스트 회귀 모델 사용을 위함
from sklearn.metrics import mean_squared_error, r2_score  # 예측 결과의 정확도 평가 지표
import matplotlib.pyplot as plt  # 결과를 그래프로 시각화
import seaborn as sns  # 시각화를 도와줌
import joblib  # 학습된 모델을 .pkl로 저장, 불러올 때 사용
import os  # 파일 경로, 디렉토리 탐색 등에 사용
```

ai-music-pppre.py(인기도 예측)에서 사용한 라이브러리
```python
import os
import numpy as np
import librosa  # 음악/오디오 파일 분석 후 음향 특성을 추출함
import joblib
from pydub import AudioSegment   # 오디오 파일을 다양한 형식(mp3, wav 등)으로 변환
import matplotlib.pyplot as plt
import matplotlib
```


