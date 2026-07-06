import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# 웹페이지 기본 설정
st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("이상 패턴 탐지 시스템")
st.markdown("정상 패턴(파란색) | 이상 패턴(빨간색) , 각 축은 압축된 잠재공간의 주요 특성을 의미")

# ==========================================
# 사이드바 (UI 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 시스템 설정")

정상_샘플수 = st.sidebar.slider("정상 유저 데이터 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 데이터 수", 10, 500, 50, 10)
탐지_민감도 = st.sidebar.slider("이상 탐지 민감도", 0.01, 0.20, 0.06, 0.01)

# ==========================================
# 데이터 생성 및 모델링
# ==========================================
np.random.seed(42)

# 가상 데이터 생성
정상_데이터 = np.random.normal(loc=[50, 10, 100, 1, 80], scale=[10, 2, 20, 0.5, 10], size=(정상_샘플수, 5))
이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5))

전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
df = pd.DataFrame(전체_데이터, columns=기본_특성)

# 이상 탐지 모델 구동
모델 = IsolationForest(n_estimators=100, contamination=탐지_민감도, random_state=42)
df['탐지결과'] = 모델.fit_predict(df[기본_특성])
df['상태'] = df['탐지결과'].map({1: '정상 패턴', -1: '이상 패턴 (차단 대상)'})

# 마커 크기 세팅
df['마커크기'] = df['상태'].map({'정상 패턴': 3, '이상 패턴 (차단 대상)': 12})

# ==========================================
# 차원 축소 및 한국어 축 정의
# ==========================================
pca = PCA(n_components=3)
잠재공간 = pca.fit_transform(df[기본_특성])

df['사용자 활동성'] = 잠재공간[:, 0]
df['결제 및 에러 패턴'] = 잠재공간[:, 1]
df['행동 변동성'] = 잠재공간[:, 2]

# ==========================================
# 메인 화면 시각화
# ==========================================
좌측_화면, 우측_화면 = st.columns([3, 1])

with 좌측_화면:
    # 3D 산점도 그래프 정의
    fig = px.scatter_3d(
        df, 
        x='사용자 활동성', 
        y='결제 및 에러 패턴', 
        z='행동 변동성', 
        color='상태',
        # 정상: 선명한 파란색(25% 투명) / 이상: 강렬한 빨간색(100% 불투명)
        color_discrete_map={
            '정상 패턴': 'rgba(10, 100, 240, 0.25)', 
            '이상 패턴 (차단 대상)': 'rgba(255, 30, 30, 1.0)'
        },
        size='마커크기',
        size_max=12,
        hover_data=기본_특성,
        template='plotly_dark'
    )
    
    # 그래프 스타일 조정
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

with 우측_화면:
    st.subheader("탐지 통계")
    
    st.metric(label="전체 모니터링 대상", value=f"{정상_샘플수 + 이상_샘플수} 명")
    이상_수 = len(df[df['상태'] == '이상 패턴 (차단 대상)'])
    st.metric(label="위험 행동 감지", value=f"{이상_수} 건", delta="- 시스템 작동중", delta_color="inverse")
    
    st.write("---")
    st.write("**⚠️ 실시간 이상 유저 목록**")
    st.dataframe(df[df['상태'] == '이상 패턴 (차단 대상)'][기본_특성].head(5))
