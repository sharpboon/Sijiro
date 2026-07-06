import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# 웹페이지 기본 설정
st.set_page_config(page_title="Fail-Safe 3D Dashboard", layout="wide")

st.title("🚀 사용자 행동 3차원 잠재공간 및 Fail-Safe 탐지기")
st.markdown("정상 패턴은 배경으로 깔고, **이상 패턴만 돋보이도록 크기와 투명도를 조절**한 3D 대시보드입니다.")

# ==========================================
# 사이드바 (UI 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 파라미터 조절")

n_normal = st.sidebar.slider("정상 유저 샘플 수", 100, 2000, 800, 100)
n_abnormal = st.sidebar.slider("이상 유저 샘플 수", 10, 500, 50, 10)
contamination = st.sidebar.slider("이상 탐지 민감도 (Contamination)", 0.01, 0.20, 0.06, 0.01)

# ==========================================
# 데이터 생성 및 모델링
# ==========================================
np.random.seed(42)

normal_users = np.random.normal(loc=[50, 10, 100, 1, 80], scale=[10, 2, 20, 0.5, 10], size=(n_normal, 5))
abnormal_users = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(n_abnormal, 5))

X_data = np.vstack([normal_users, abnormal_users])
feature_names = ['Duration', 'Clicks', 'Amount', 'Errors', 'Scroll_Depth']
df = pd.DataFrame(X_data, columns=feature_names)

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[feature_names])
df['Status'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly (Fail-Safe)'})

# 가독성을 위한 마커 크기 세팅 (정상: 작게, 이상: 크게)
df['Marker_Size'] = df['Status'].map({'Normal': 3, 'Anomaly (Fail-Safe)': 12})

# 3. PCA 3차원 차원 축소
pca = PCA(n_components=3)
latent_3d = pca.fit_transform(df[feature_names])
df['Latent_X'] = latent_3d[:, 0]
df['Latent_Y'] = latent_3d[:, 1]
df['Latent_Z'] = latent_3d[:, 2]

# ==========================================
# 대시보드 메인 화면 3D 시각화
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    # Plotly 3D 산점도 (가독성 최적화)
    fig = px.scatter_3d(
        df, x='Latent_X', y='Latent_Y', z='Latent_Z', 
        color='Status',
        # RGBA 색상을 사용하여 투명도를 개별적으로 적용 (정상은 20% 투명도, 이상은 100% 불투명)
        color_discrete_map={
            'Normal': 'rgba(0, 204, 150, 0.2)', 
            'Anomaly (Fail-Safe)': 'rgba(239, 85, 59, 1.0)'
        },
        size='Marker_Size',
        size_max=12,
        hover_data=feature_names,
        template='plotly_dark' # 다크 테마 적용으로 가독성 폭발
    )
    
    # 3D 축 배경 및 그리드 라인 깔끔하게 정리
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='gray'),
            yaxis=dict(showbackground=False, gridcolor='gray'),
            zaxis=dict(showbackground=False, gridcolor='gray')
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True, height=700)

with col2:
    st.subheader("📊 탐지 요약")
    normal_count = len(df[df['Status'] == 'Normal'])
    anomaly_count = len(df[df['Status'] == 'Anomaly (Fail-Safe)'])
    
    st.metric(label="총 유저 수", value=f"{n_normal + n_abnormal} 명")
    st.metric(label="탐지된 이상 패턴 수", value=f"{anomaly_count} 명", delta="- 차단 권장", delta_color="inverse")
    
    st.write("---")
    st.write("**이상 유저 데이터 미리보기**")
    st.dataframe(df[df['Status'] == 'Anomaly (Fail-Safe)'][feature_names].head(10))
