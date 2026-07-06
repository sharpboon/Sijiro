import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("🚀 사용자 행동 3차원 잠재공간 및 이상 패턴 탐지 시스템")

# ==========================================
# 시나리오별 기본 데이터 정의
# ==========================================
SCENARIO_DEFAULTS = {
    "1. 평범한 평일 (기본)": {"체류시간": 50, "클릭수": 10, "결제액": 100, "에러수": 1, "스크롤깊이": 80},
    "2. 주말 대규모 이벤트 (활동/결제 폭주)": {"체류시간": 90, "클릭수": 30, "결제액": 400, "에러수": 2, "스크롤깊이": 150},
    "3. 서버 장애 발생 (에러 폭주)": {"체류시간": 20, "클릭수": 5, "결제액": 10, "에러수": 25, "스크롤깊이": 30}
}

# ==========================================
# 세션 상태(Session State) 초기화 로직
# ==========================================
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "1. 평범한 평일 (기본)"

# 최초 실행 시 기본값 세팅
for key, val in SCENARIO_DEFAULTS[st.session_state.current_scenario].items():
    if f"cfg_{key}" not in st.session_state:
        st.session_state[f"cfg_{key}"] = val

# ==========================================
# 사이드바 (UI 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 시스템 설정")

st.sidebar.markdown("### 🎲 데이터 시뮬레이션")
if st.sidebar.button("🔄 새로운 무작위 유저 패턴 생성", use_container_width=True):
    st.session_state.random_seed = np.random.randint(0, 100000)

시나리오 = st.sidebar.selectbox(
    "시나리오 환경 선택", 
    list(SCENARIO_DEFAULTS.keys()),
    index=list(SCENARIO_DEFAULTS.keys()).index(st.session_state.current_scenario)
)

if 시나리오 != st.session_state.current_scenario:
    st.session_state.current_scenario = 시나리오
    for key, val in SCENARIO_DEFAULTS[시나리오].items():
        st.session_state[f"cfg_{key}"] = val
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ 정상치 기준값(평균) 수동 조절")

if st.sidebar.button("↩️ 현재 시나리오 기본값으로 초기화", use_container_width=True):
    for key, val in SCENARIO_DEFAULTS[시나리오].items():
        st.session_state[f"cfg_{key}"] = val
    st.rerun()

정상_체류 = st.sidebar.slider("정상 체류시간 평균", 0, 200, key="cfg_체류시간")
정상_클릭 = st.sidebar.slider("정상 클릭수 평균", 0, 100, key="cfg_클릭수")
정상_결제 = st.sidebar.slider("정상 결제액 평균", 0, 1000, key="cfg_결제액")
정상_에러 = st.sidebar.slider("정상 에러수 평균", 0, 50, key="cfg_에러수")
정상_스크롤 = st.sidebar.slider("정상 스크롤깊이 평균", 0, 300, key="cfg_스크롤깊이")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 모니터링 데이터 규모")
정상_샘플수 = st.sidebar.slider("일반 유저 데이터 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 데이터 수", 10, 500, 50, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 판정 민감도 (상위 % 기준)")
위험_비율 = st.sidebar.slider("🔴 위험(차단) 판정 비율", min_value=1, max_value=20, value=6, step=1, format="%d%%")
주의_비율 = st.sidebar.slider("🟡 주의(관찰) 판정 비율", min_value=5, max_value=30, value=10, step=1, format="%d%%")

탐지_민감도 = 위험_비율 / 100.0
주의_분위수 = 1.0 - (주의_비율 / 100.0)

# ==========================================
# 데이터 생성 및 모델링 + (신규) 고유 유저 번호 부여
# ==========================================
np.random.seed(st.session_state.random_seed)

정상_기준점 = [정상_체류, 정상_클릭, 정상_결제, 정상_에러, 정상_스크롤]
정상_데이터 = np.random.normal(loc=정상_기준점, scale=[10, 2, 20, 0.5, 10], size=(정상_샘플수, 5))
이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5))

전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
df = pd.DataFrame(전체_데이터, columns=기본_특성)
df[기본_특성] = df[기본_특성].clip(lower=0) 

# 고유 유저 번호 칼럼 주입 (예: USR-0001)
df.insert(0, '유저 번호', [f"USR-{i+1:04d}" for i in range(len(df))])

# 모델 구동
모델 = IsolationForest(n_estimators=100, contamination=탐지_민감도, random_state=st.session_state.random_seed)
df['기본탐지'] = 모델.fit_predict(df[기본_특성])

내부_점수 = 모델.decision_function(df[기본_특성])
df['위험도 점수'] = np.round(-내부_점수 * 100, 1)

# ==========================================
# 3단계 상태 분류 로직
# ==========================================
주의_기준점 = df[df['기본탐지'] == 1]['위험도 점수'].quantile(주의_분위수)

def classify_status(row):
    if row['기본탐지'] == -1:
        return '🔴 위험 (차단 대상)'
    elif row['위험도 점수'] >= 주의_기준점:
        return '🟡 주의 (관찰 요망)'
    else:
        return '🔵 안전 (정상 패턴)'

df['상태'] = df.apply(classify_status, axis=1)
df['마커크기'] = df['상태'].map({'🔵 안전 (정상 패턴)': 2, '🟡 주의 (관찰 요망)': 6, '🔴 위험 (차단 대상)': 15})

# ==========================================
# 주요 이상 원인 파악 및 색상 강조
# ==========================================
정상_평균 = df[df['상태'] == '🔵 안전 (정상 패턴)'][기본_특성].mean()
정상_표준편차 = df[df['상태'] == '🔵 안전 (정상 패턴)'][기본_특성].std().replace(0, 1)

표준점수 = np.abs((df[기본_특성] - 정상_평균) / 정상_표준편차)
df['주요원인_특성'] = 표준점수.idxmax(axis=1)

def format_hover_text(row, col):
    값 = row[col]
    편차 = 값 - 정상_평균[col]
    편차표시 = f"+{편차:.1f}" if 편차 > 0 else f"{편차:.1f}"
    기본문구 = f"{값:.1f} (기준대비 {편차표시})"
    
    if row['상태'] != '🔵 안전 (정상 패턴)' and row['주요원인_특성'] == col:
        if '🔴 위험' in row['상태']:
            return f'<b><span style="color:#D32F2F;">{기본문구} ◀ 원인</span></b>'
        else:
            return f'<b><span style="color:#E65100;">{기본문구} ◀ 원인</span></b>'
    return 기본문구

for 특성 in 기본_특성:
    df[f'{특성}_표시'] = df.apply(lambda row: format_hover_text(row, 특성), axis=1)

# ==========================================
# 차원 축소 (PCA)
# ==========================================
pca = PCA(n_components=3)
잠재공간 = pca.fit_transform(df[기본_특성])
df['잠재축 X'] = 잠재공간[:, 0]
df['잠재축 Y'] = 잠재공간[:, 1]
df['잠재축 Z'] = 잠재공간[:, 2]

# ==========================================
# 사용자 안내 가이드 레이아웃
# ==========================================
with st.expander("💡 실시간 등급 판정 및 데이터별 이상 원인 분석 기준 안내", expanded=True):
    공지_좌, 공지_우 = st.columns(2)
    with 공지_좌:
        st.markdown(f"""
        #### 📌 3단계 탐지 등급 산정 기준
        * **🔵 안전 (정상 패턴)**: 설정된 정상치 기준값 근처에 밀집된 안정적인 일반 유저군입니다.
        * **🟡 주의 (관찰 요망)**: 수동 설정된 기준에서 살짝 벗어나 위험도 점수가 높은 **상위 {주의_비율}%**의 경계선 데이터입니다.
        * **🔴 위험 (차단 대상)**: AI 모델이 현재 기준점과 명백히 다른 패턴으로 탐지한 **상위 {위험_비율}%**의 고위험군 유저입니다.
        """)
    with 공지_우:
        st.markdown("""
        #### 🔍 데이터별 동적 원인(`◀ 원인`) 도출 방식
        * 운영자가 사이드바에서 **정상치 기준값을 수정할 때마다 AI가 새로운 기준을 바탕으로 학습**을 즉시 다시 수행합니다.
        * 각 데이터가 변경된 **'실시간 정상 평균 지표'에서 얼마나 멀리 벗어났는지**를 통계적으로 추적합니다.
        * 5개 지표 중 기준 점수와 가장 괴리가 큰 핵심 지표 하나를 `◀ 원인` 문구와 색상으로 강조합니다.
        """)

# ==========================================
# 메인 화면 시각화 (유저 번호 툴팁 연동)
# ==========================================
좌측_화면, 우측_화면 = st.columns([3, 1.2]) 

with 좌측_화면:
    fig = px.scatter_3d(
        df, x='잠재축 X', y='잠재축 Y', z='잠재축 Z', 
        color='상태',
        color_discrete_map={
            '🔵 안전 (정상 패턴)': 'rgba(30, 136, 229, 0.15)', 
            '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.8)',   
            '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'    
        },
        size='마커크기', size_max=15,  # <-- 치명적 오타(마кер크기) 수정 완료!
        custom_data=['유저 번호', '상태', '위험도 점수', '체류시간_표시', '클릭수_표시', '결제액_표시', '에러수_표시', '스크롤깊이_표시'],
        template='plotly_dark'
    )
    
    fig.update_traces(
        hovertemplate=(
            "<b>[%{customdata[0]}] %{customdata[1]}</b><br>"
            "위험도 점수: %{customdata[2]}<br>"
            "-----------------------------------<br>"
            "체류시간 : %{customdata[3]}<br>"
            "클릭수   : %{customdata[4]}<br>"
            "결제액   : %{customdata[5]}<br>"
            "에러수   : %{customdata[6]}<br>"
            "스크롤   : %{customdata[7]}<br>"
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

# ==========================================
# 우측 화면 통계 및 개별 유저 검색창
# ==========================================
with 우측_화면:
    st.subheader("📊 탐지 통계")
    
    안전_수 = len(df[df['상태'] == '🔵 안전 (정상 패턴)'])
    주의_수 = len(df[df['상태'] == '🟡 주의 (관찰 요망)'])
    위험_수 = len(df[df['상태'] == '🔴 위험 (차단 대상)'])
    
    통계_좌, 통계_우 = st.columns(2)
    with 통계_좌:
        st.metric(label="전체 모니터링 대상", value=f"{len(df)} 명")
        st.metric(label="🟡 주의 행동 감지", value=f"{주의_수} 건")
    with 통계_우:
        st.metric(label="🔴 위험 행동 감지", value=f"{위험_수} 건")
        
    st.markdown("---")
    
    # 개별 유저 실시간 검색 섹션
    st.subheader("🔍 개별 유저 상세 검색")
    선택된_유저 = st.selectbox(
        "유저 번호를 선택하거나 타이핑하여 검색하세요.",
        options=["선택 안 함"] + list(df['유저 번호'].values),
        index=0
    )
    
    if 선택된_유저 != "선택 안 함":
        유저_데이터 = df[df['유저 번호'] == 선택된_유저].iloc[0]
        
        # 유저 정보 요약 카드 출력
        with st.container(border=True):
            st.markdown(f"### 👤 {선택된_유저} 프로필")
            st.markdown(f"**현재 상태:** {유저_데이터['상태']}")
            st.markdown(f"**위험도 점수:** `{유저_데이터['위험도 점수']}` / 100 점")
            
            if 유저_데이터['상태'] != '🔵 안전 (정상 패턴)':
                st.markdown(f"⚠️ **주요 특이 원인:** <span style='color:#FF5722; font-weight:bold;'>{유저_데이터['주요원인_특성']}</span>", unsafe_allow_html=True)
            
            st.markdown("**📋 상세 지표 현황**")
            지표_데이터 = {}
            for 특성 in 기본_특성:
                편차 = 유저_데이터[특성] - 정상_평균[특성]
                편차_문구 = f"+{편차:.1f}" if 편차 > 0 else f"{편차:.1f}"
                지표_데이터[특성] = f"{유저_데이터[특성]:.1f} (정상대비 {편차_문구})"
            st.json(지표_데이터)
            
    st.markdown("---")
    st.write("**⚠️ 실시간 고위험 유저 (Top 10)**")
    
    보여줄_컬럼 = ['유저 번호', '상태', '위험도 점수'] + 기본_특성
    st.dataframe(
        df[df['상태'] == '🔴 위험 (차단 대상)'][보여줄_컬럼]
        .sort_values(by='위험도 점수', ascending=False)
        .head(10),
        hide_index=True
    )

