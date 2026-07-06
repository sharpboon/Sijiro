import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# 웹페이지 기본 설정 (넓은 화면 사용)
st.set_page_config(page_title="Fail-Safe 3D Dashboard", layout="wide")

st.title("🚀 사용자 행동 3차원 잠재공간 및 Fail-Safe 탐지기")
st.markdown("좌측 사이드바에서 값을 조절하면 **실시간으로 3D 그래프와 이상 탐지 결과가 업데이트** 됩니다.")

# ==========================================
# 사이드바 (UI 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 파라미터 조절")

st.sidebar.subheader("데이터 설정")
n_normal = st.sidebar.slider("정상 유저 샘플 수", min_value=100, max_value=2000, value=800, step=100)
n_abnormal = st.sidebar.slider("이상 유저 샘플 수", min_value=10, max_value=500, value=50, step=10)

st.sidebar.subheader("모델 설정")
contamination = st.sidebar.slider("이상 탐지 민감도 (Contamination)", min_value=0.01, max_value=0.20, value=0.06, step=0.01)

# ==========================================
# 데이터 생성 및 모델링 (실시간 업데이트)
# ==========================================
np.random.seed(42)

# 1. 데이터 생성
normal_users = np.random.normal(loc=[50, 10, 100, 1, 80], scale=[10, 2, 20, 0.5, 10], size=(n_normal, 5))
abnormal_users = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(n_abnormal, 5))

X_data = np.vstack([normal_users, abnormal_users])
feature_names = ['Duration', 'Clicks', 'Amount', 'Errors', 'Scroll_Depth']
df = pd.DataFrame(X_data, columns=feature_names)

# 2. Isolation Forest를 이용한 Fail-Safe
iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[feature_names])
df['Status'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly (Fail-Safe)'})

# 3. PCA 3차원 차원 축소
pca = PCA(n_components=3)
latent_3d = pca.fit_transform(df[feature_names])
df['Latent_X'] = latent_3d[:, 0]
df['Latent_Y'] = latent_3d[:, 1]
df['Latent_Z'] = latent_3d[:, 2]

# ==========================================
# 대시보드 메인 화면 시각화
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    # Plotly 3D 산점도
    fig = px.scatter_3d(
        df, x='Latent_X', y='Latent_Y', z='Latent_Z', color='Status',
        color_discrete_map={'Normal': '#00CC96', 'Anomaly (Fail-Safe)': '#EF553B'},
        opacity=0.8,
        hover_data=feature_names
    )
    fig.update_layout(
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
