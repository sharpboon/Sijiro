import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("🚀 사용자 행동 3차원 잠재공간 및 이상 패턴 탐지 시스템")
st.markdown("데이터를 **🔵 안전, 🟡 주의, 🔴 위험** 3단계로 세분화하여 경계선의 애매한 패턴을 걸러냅니다.")

# ==========================================
# 사이드바 (UI 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 시스템 설정")
정상_샘플수 = st.sidebar.slider("정상 유저 데이터 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 데이터 수", 10, 500, 50, 10)
탐지_민감도 = st.sidebar.slider("이상 탐지 민감도 (위험 기준)", 0.01, 0.20, 0.06, 0.01)

# ==========================================
# 데이터 생성 및 모델링
# ==========================================
np.random.seed(42)

정상_데이터 = np.random.normal(loc=[50, 10, 100, 1, 80], scale=[10, 2, 20, 0.5, 10], size=(정상_샘플수, 5))
이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5))

전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
df = pd.DataFrame(전체_데이터, columns=기본_특성)

모델 = IsolationForest(n_estimators=100, contamination=탐지_민감도, random_state=42)
df['기본탐지'] = 모델.fit_predict(df[기본_특성])

내부_점수 = 모델.decision_function(df[기본_특성])
df['위험도 점수'] = np.round(-내부_점수 * 100, 1)

# ==========================================
# (핵심) 3단계 상태 분류 로직 적용
# ==========================================
# 정상 범주에 속한 데이터 중, 위험도가 가장 높은 상위 10%를 '주의'로 분류
주의_기준점 = df[df['기본탐지'] == 1]['위험도 점수'].quantile(0.90)

def classify_status(row):
    if row['기본탐지'] == -1:
        return '🔴 위험 (차단 대상)'
    elif row['위험도 점수'] >= 주의_기준점:
        return '🟡 주의 (관찰 요망)'
    else:
        return '🔵 안전 (정상 패턴)'

df['상태'] = df.apply(classify_status, axis=1)

# 마커 크기 세팅 (안전: 제일 작게, 주의: 중간, 위험: 가장 크게)
df['마커크기'] = df['상태'].map({
    '🔵 안전 (정상 패턴)': 2, 
    '🟡 주의 (관찰 요망)': 6, 
    '🔴 위험 (차단 대상)': 15
})

# ==========================================
# 정상 평균 대비 차이값 계산
# ==========================================
정상_평균 = df[df['상태'] == '🔵 안전 (정상 패턴)'][기본_특성].mean()

for 특성 in 기본_특성:
    편차 = df[특성] - 정상_평균[특성]
    df[f'{특성}_표시'] = (
        df[특성].round(1).astype(str) + 
        " (정상평균 대비 " + 
        편차.round(1).apply(lambda x: f"+{x}" if x > 0 else f"{x}") + 
        ")"
    )

# ==========================================
# 차원 축소
# ==========================================
pca = PCA(n_components=3)
잠재공간 = pca.fit_transform(df[기본_특성])

df['잠재축 X'] = 잠재공간[:, 0]
df['잠재축 Y'] = 잠재공간[:, 1]
df['잠재축 Z'] = 잠재공간[:, 2]

# ==========================================
# 메인 화면 시각화
# ==========================================
좌측_화면, 우측_화면 = st.columns([3, 1])

with 좌측_화면:
    fig = px.scatter_3d(
        df, 
        x='잠재축 X', y='잠재축 Y', z='잠재축 Z', 
        color='상태',
        color_discrete_map={
            '🔵 안전 (정상 패턴)': 'rgba(30, 136, 229, 0.15)', # 파란색 반투명
            '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.8)',   # 노란색 불투명
            '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'    # 빨간색 불투명
        },
        size='마커크기',
        size_max=15,
        custom_data=['상태', '위험도 점수', '체류시간_표시', '클릭수_표시', '결제액_표시', '에러수_표시', '스크롤깊이_표시'],
        template='plotly_dark'
    )
    
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "위험도 점수: %{customdata[1]}<br>"
            "-----------------------------------<br>"
            "체류시간 : %{customdata[2]}<br>"
            "클릭수   : %{customdata[3]}<br>"
            "결제액   : %{customdata[4]}<br>"
            "에러수   : %{customdata[5]}<br>"
            "스크롤   : %{customdata[6]}<br>"
            "<extra></extra>"
        )
    )
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=13, font_color="black", bordercolor="black"),
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='주요 활동성 (X)'),
            yaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='결제/에러 (Y)'),
            zaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='행동 변동성 (Z)')
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True, height=700)

with 우측_화면:
    st.subheader("📊 탐지 통계")
    
    안전_수 = len(df[df['상태'] == '🔵 안전 (정상 패턴)'])
    주의_수 = len(df[df['상태'] == '🟡 주의 (관찰 요망)'])
    위험_수 = len(df[df['상태'] == '🔴 위험 (차단 대상)'])
    
    st.metric(label="전체 모니터링 대상", value=f"{len(df)} 명")
    st.metric(label="🔴 위험 행동 감지", value=f"{위험_수} 건")
    st.metric(label="🟡 주의 행동 감지", value=f"{주의_수} 건")
    
    st.write("---")
    st.write("**⚠️ 실시간 위험 유저 (Top 10)**")
    
    보여줄_컬럼 = ['상태', '위험도 점수'] + 기본_특성
    st.dataframe(
        df[df['상태'] == '🔴 위험 (차단 대상)'][보여줄_컬럼]
        .sort_values(by='위험도 점수', ascending=False)
        .head(10)
    )
