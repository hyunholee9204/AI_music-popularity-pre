import os
import numpy as np
import librosa
import joblib
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib

# ——— 한글 폰트 설정 ———
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 1. 모델 로드
model_path = r'C:\Users\hyeon\OneDrive\바탕 화면\데이터 분석 프로젝트\스포티파이 데이터 분석\popularity_model.pkl'
model = joblib.load(model_path)

# 2. MP3 파일이 있는 폴더 경로
folder_path = r'C:\Users\hyeon\OneDrive\바탕 화면\데이터 분석 프로젝트\스포티파이 데이터 분석\ai 노래'

# 3. 오디오 특성 추출 함수 (8개 특성, 오류시 0.0 대체)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"특성 추출 실패—librosa.load(): {e}")
        return None

    def safe_extract(name, func):
        try:
            val = np.mean(func())
            print(f"{name} 추출 성공: {val}")
            return val
        except Exception as e:
            print(f"{name} 추출 오류: {e}")
            return 0.0

    valence       = safe_extract("valence",       lambda: librosa.feature.chroma_stft(y=y, sr=sr))
    energy        = safe_extract("energy",        lambda: librosa.feature.rms(y=y))
    danceability  = safe_extract("danceability",  lambda: librosa.feature.zero_crossing_rate(y))
    # 안정적인 tempo 추출
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo     = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
        print(f"tempo 추출 성공: {tempo}")
    except Exception as e:
        print(f"tempo 추출 오류: {e}")
        tempo = 0.0

    duration_ms   = len(y) / sr * 1000
    speechiness   = safe_extract("speechiness",   lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr))
    acousticness  = safe_extract("acousticness",  lambda: librosa.feature.spectral_flatness(y=y))
    instrumentalness = safe_extract("instrumentalness", lambda: librosa.feature.spectral_contrast(y=y, sr=sr))

    return np.array([[valence, energy, danceability, tempo,
                      duration_ms, speechiness, acousticness, instrumentalness]])


# 4. 예측 결과 저장
results = []

# 5. MP3 순회→변환→특성→예측
for filename in os.listdir(folder_path):
    if not filename.lower().endswith('.mp3'):
        continue

    print(f"\n파일 처리 중: {filename}")
    mp3_path    = os.path.join(folder_path, filename)
    temp_wav    = os.path.join(folder_path, 'temp.wav')

    try:
        # MP3→WAV
        AudioSegment.from_mp3(mp3_path).export(temp_wav, format='wav')

        # 특성 추출
        features = extract_features(temp_wav)
        if features is None:
            raise ValueError("특성 추출 실패")

        print(f"features shape: {features.shape}")

        # 예측
        pred = model.predict(features)[0]
        print(f"예측 완료: popularity = {pred:.2f}")
        results.append((filename, round(pred, 2)))

    except Exception as e:
        print(f"예측 실패: {e}")
        results.append((filename, f"오류"))


# 6. 콘솔 출력
print("\n=== 예측 결과 ===")
for name, score in results:
    print(f"{name} → {score}")


# 7. 시각화 (정상 예측만)
valid = [(n, s) for n, s in results if isinstance(s, (int, float, np.floating))]
if valid:
    names, scores = zip(*valid)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores)
    plt.title('AI 노래별 예측 Popularity 점수', fontsize=14)
    plt.xlabel('노래 제목')
    plt.ylabel('예측 Popularity (0~100)')
    plt.ylim(0, 100)
    plt.xticks(rotation=15)

    for bar, sc in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                 f'{sc}', ha='center')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("유효한 예측 결과가 없어 시각화를 건너뜁니다.")
