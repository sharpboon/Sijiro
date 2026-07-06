import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# 웹페이지 기본 설정
st.set_page_config(page_title="Fail-Safe 3D Dashboard", layout="wide")

st.title("Fail-Safe 탐지기")

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
feature_names = ['체류시간', '클릭수', '결제액', '에러수', '스크롤']
df = pd.DataFrame(X_data, columns=feature_names)

# Isolation Forest를 이용한 이상 탐지
iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[feature_names])
df['Status'] = df['Anomaly_Score'].map({1: '정상', -1: '이상'})

# 가독성을 위한 마커 크기 세팅
df['Marker_Size'] = df['Status'].map({'정상': 3, '이상': 12})

# ==========================================
# 3. PCA 3차원 차원 축소 및 직관적 네이밍
# ==========================================
pca = PCA(n_components=3)
latent_3d = pca.fit_transform(df[feature_names])

# X, Y, Z 대신 실무적으로 이해하기 쉬운 잠재공간 이름 부여
df['잠재축 X (주요 활동성)'] = latent_3d[:, 0]
df['잠재축 Y (결제/에러 패턴)'] = latent_3d[:, 1]
df['잠재축 Z (행동 변동성)'] = latent_3d[:, 2]

# ==========================================
# 대시보드 메인 화면 3D 시각화
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    # Plotly 3D 산점도
    fig = px.scatter_3d(
        df, 
        x='주요 활동성', 
        y='결제/에러 패턴', 
        z='행동 변동성', 
        color='Status',
        # 정상 유저: 눈에 잘 띄는 파란색(Blue) 반투명 처리 / 이상 유저: 강렬한 빨간색(Red) 불투명
        color_discrete_map={
            '정상': 'rgba(30, 136, 229, 0.3)', 
            '이상': 'rgba(255, 50, 50, 1.0)'
        },
        size='Marker_Size',
        size_max=12,
        hover_data=feature_names, # 원본 데이터(체류시간 등)도 마우스 올리면 보이게 유지
        template='plotly_dark'
    )
    
    # 3D 축 배경 지우기 및 라인 세팅
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
    st.subheader("탐지 요약")
    normal_count = len(df[df['Status'] == '정상'])
    anomaly_count = len(df[df['Status'] == '이상'])
    
    st.metric(label="총 모니터링 유저", value=f"{n_normal + n_abnormal} 명")
    st.metric(label="탐지된 이상 패턴", value=f"{anomaly_count} 건", delta="- Fail-Safe 작동", delta_color="inverse")
    
    st.write("---")
    st.write("**⚠️ 이상 탐지 데이터 (Top 5)**")
    st.dataframe(df[df['Status'] == 'Anomaly (이상/차단대상)'][feature_names].head(5))
