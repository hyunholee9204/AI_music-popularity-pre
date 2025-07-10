import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# CSV 파일 불러오기
df = pd.read_csv(r'C:\Users\hyeon\OneDrive\바탕 화면\데이터 분석 프로젝트\스포티파이 데이터 분석\universal_top_spotify_songs.csv', low_memory=False)

# 날짜 변환 및 연도 필터링
df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
df['year'] = df['snapshot_date'].dt.year
df_recent = df[(df['year'] >= 2020) & (df['year'] <= 2025)]

# 사용할 feature 목록
features = [
    'valence', 'energy', 'danceability', 'tempo',
    'duration_ms', 'speechiness', 'acousticness', 'instrumentalness'
]

# 결측치 제거
df_model = df_recent[features + ['popularity']].dropna()

# 입력 X, 목표값 y 분리
X = df_model[features]
y = df_model['popularity']

# 학습용 / 테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score (설명력): {r2:.2f}")

# Feature 중요도 시각화
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature 중요도 (popularity 예측)', fontsize=14)
plt.xlabel('중요도')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.show()

# 예측 결과 일부 출력
result_df = pd.DataFrame({
    '실제 popularity': y_test.values,
    '예측 popularity': y_pred
}).reset_index(drop=True)
print(result_df.head(10))

# 모델 저장
save_path = os.path.join(os.getcwd(), 'popularity_model.pkl')
joblib.dump(model, save_path)
print(f"✅ 모델 저장 완료: {save_path}")
