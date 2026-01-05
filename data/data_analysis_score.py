!pip install sentence-transformers statsmodels pandas scikit-learn openpyxl wordcloud
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sentence_transformers import SentenceTransformer, util
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import os

# ==============================================================================
# [핵심 설정] 폰트 강제 주입 (한글 깨짐 방지)
# ==============================================================================
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 한글 폰트 설정 완료: {font_name}")
else:
    print("❌ 폰트 파일을 찾을 수 없습니다. 시각화 시 한글이 깨질 수 있습니다.")

# ==============================================================================
# 1. 데이터 로드 및 전처리 (수정됨: CSV 파일 로드)
# ==============================================================================
file_name = 'youtube_data_filtered_final.csv'  # 파일명 변경
print(f"📂 '{file_name}' 로드 중...")

try:
    # CSV 파일 읽기 (한글 깨짐 방지를 위해 인코딩 자동 처리 시도)
    try:
        df = pd.read_csv(file_name, encoding='utf-8-sig')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 인코딩 실패, CP949로 재시도합니다...")
        df = pd.read_csv(file_name, encoding='cp949')

except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다.")
    df = pd.DataFrame()

if not df.empty:
    # 숫자 변환 및 텍스트 정제 (기존 코드 유지)
    numeric_cols = ['조회수', '구독자수', '채널평균조회수', '영상길이(초)', '업로드일수']
    for col in numeric_cols:
        if col in df.columns:
            # 쉼표(,) 제거 후 숫자 변환 처리 추가 (CSV는 텍스트로 읽힐 수 있음)
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    stop_words = ['브이로그', 'vlog', 'ep', '편', '화', 'video', 'full', 'hd']
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        for w in stop_words: text = text.replace(w, '')
        return text.strip()

    df['cleaned_title'] = df['영상제목'].apply(clean_text)

    # 2. 주제별 유사도 및 회귀분석 수행 (기존 코드 유지)
    print("⏳ 주제별 유사도 계산 및 회귀분석 준비 중...")
    model = SentenceTransformer('jhgan/ko-sbert-multitask')
    target_topics = ['일상', '여행', '먹방']
    title_emb = model.encode(df['cleaned_title'].tolist(), convert_to_tensor=True)

    topic_score_cols = []
    for topic in target_topics:
        topic_emb = model.encode(topic, convert_to_tensor=True)
        col_name = f'Score_{topic}'
        scores = util.cos_sim(title_emb, topic_emb).cpu().numpy().flatten()
        df[col_name] = scores * 100
        topic_score_cols.append(col_name)

    df['ln_views'] = np.log1p(df['조회수'])
    x_numeric_cols = ['구독자수', '채널평균조회수', '영상길이(초)', '업로드일수']
    final_x_cols = []

    for col in x_numeric_cols:
        if col in df.columns:
            df[f'ln_{col}'] = np.log1p(df[col])
            final_x_cols.append(f'ln_{col}')
    final_x_cols.extend(topic_score_cols)

    X = df[final_x_cols]
    X = sm.add_constant(X)
    y = df['ln_views']

    # 결측치가 있으면 제거하고 모델링 (CSV 특성상 안전장치 추가)
    if X.isnull().values.any() or y.isnull().values.any():
        print("⚠️ 데이터에 결측치가 있어 제거 후 분석합니다.")
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        df = df.loc[valid_idx]

    model_ols = sm.OLS(y, X).fit()

    # 회귀분석 결과표 출력
    print("\n" + "="*60)
    print("📊 [OLS 회귀분석 결과표]")
    print("="*60)
    print(model_ols.summary())
    print("="*60 + "\n")


    # ==============================================================================
    # 3. 썸네일 점수 계산 (변경) - 0점~100점 스케일 적용
    # ==============================================================================
    print("⭐ 썸네일 점수 0~100점 스케일링 중...")

    # 1) 잔차 계산 (통제 변수로 설명되지 않은 초과 성과)
    df['residual'] = model_ols.resid

    # 2) Min-Max Scaling을 통해 잔차를 0~100으로 변환
    resid_min = df['residual'].min()
    resid_max = df['residual'].max()

    if resid_max != resid_min:
        # 공식: (현재 잔차 - 최소 잔차) / (최대 잔차 - 최소 잔차) * 100
        df['Thumbnail_Score'] = (df['residual'] - resid_min) / (resid_max - resid_min) * 100
    else:
        # 모든 잔차가 같을 경우 (극히 드뭄)
        df['Thumbnail_Score'] = 50.0

    # --------------------------------------------------------------------------
    # 4. 결과 저장 (원본 순서 유지)
    # --------------------------------------------------------------------------
    output_filename = '썸네일_분석결과_원본순서(0-100점).xlsx' # 결과는 엑셀로 유지
    df.to_excel(output_filename, index=False)
    print(f"📄 분석 결과 저장 완료! 👉 '{output_filename}' 파일을 확인해보세요.")


    # ==============================================================================
    # 5. 시각화 (0~100점 스케일로 표시)
    # ==============================================================================
    print("\n🎨 시각화 자료 생성 중...")

    # 폰트 속성 객체 정의
    prop = fm.FontProperties(fname=font_path, size=12)
    prop_title = fm.FontProperties(fname=font_path, size=16, weight='bold')

    # (1) 워드클라우드
    def plot_wordclouds(df, topics):
        wc_stop_words = set(['브이로그', 'vlog', 'ep', '편', '화', '영상', 'video', 'full', 'hd', '진짜', '너무', '오늘', '하는'])
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, topic in enumerate(topics):
            col_name = f'Score_{topic}'
            if col_name in df.columns:
                top_videos = df.nlargest(30, col_name)
                text_list = top_videos['cleaned_title'].tolist()
                combined_text = ' '.join(str(t) for t in text_list)
                words = re.findall(r'[가-힣]{2,}', combined_text)
                words = [w for w in words if w not in wc_stop_words]

                if not words:
                    axes[i].text(0.5, 0.5, "데이터 부족", ha='center', fontproperties=prop)
                else:
                    wc = WordCloud(font_path=font_path, background_color='white', width=600, height=400, colormap='viridis', max_words=50)
                    wc.generate(' '.join(words))
                    axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f"'{topic}' 상위 키워드", fontsize=16, fontweight='bold', fontproperties=prop_title)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    # (2) 잔차 및 랭킹 차트
    def plot_thumbnail_performance(df, model):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # [왼쪽] 잔차 시각화
        pred_val = model.predict()
        true_val = df['ln_views']

        # 썸네일 점수가 0~100이므로 cmap=coolwarm_r 그대로 사용
        sc = axes[0].scatter(pred_val, true_val, c=df['Thumbnail_Score'],
                             cmap='coolwarm_r', alpha=0.7, edgecolors='w', s=60)

        min_v = min(pred_val.min(), true_val.min())
        max_v = max(pred_val.max(), true_val.max())
        axes[0].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, label='예측=실제')

        axes[0].set_title('썸네일 성과(잔차) 분포도', fontproperties=prop_title)
        axes[0].set_xlabel(' 회귀 분석 조회수 (Log)', fontproperties=prop)
        axes[0].set_ylabel('실제 조회수 (Log)', fontproperties=prop)
        axes[0].legend(prop=prop)

        cbar = plt.colorbar(sc, ax=axes[0])
        cbar.set_label('썸네일 점수 (0~100점)', fontproperties=prop)

        # [오른쪽] 랭킹 차트
        top5 = df.nlargest(5, 'Thumbnail_Score')
        bottom5 = df.nsmallest(5, 'Thumbnail_Score')
        rank_df = pd.concat([top5, bottom5]).sort_values('Thumbnail_Score')

        # 50점을 기준으로 색상 구분
        colors = ['#4ecdc4' if x < 50 else '#ff6b6b' for x in rank_df['Thumbnail_Score']]

        axes[1].barh(range(len(rank_df)), rank_df['Thumbnail_Score'], color=colors)
        axes[1].set_yticks(range(len(rank_df)))
        axes[1].set_yticklabels([str(t)[:10] + '...' for t in rank_df['영상제목']], fontproperties=prop)

        axes[1].axvline(50, color='black', linewidth=0.8, linestyle='--') # 기준선을 50점으로 변경
        axes[1].set_title('썸네일/기획 성과 Best 5 vs Worst 5', fontproperties=prop_title)
        axes[1].set_xlabel('성과 점수 (0~100점)', fontproperties=prop)
        axes[1].set_xlim(0, 100) # x축 범위 0~100 고정

        plt.tight_layout()
        plt.show()

    # 시각화 실행
    plot_wordclouds(df, target_topics)
    plot_thumbnail_performance(df, model_ols)

    print("\n🎉 모든 분석 및 시각화 완료! (썸네일 점수 0~100점 반영)")
