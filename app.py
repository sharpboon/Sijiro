import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("🚀 사용자 행동 3차원 잠재공간 및 이상 패턴 탐지 시스템")

# ==========================================
# 시나리오별 고정 데이터 표준 정의
# ==========================================
SCENARIO_DEFAULTS = {
    "1. 평범한 평일 (기본)": {"체류시간": 50, "클릭수": 10, "결제액": 100, "에러수": 1, "스크롤깊이": 80},
    "2. 주말 대규모 이벤트 (활동/결제 폭주)": {"체류시간": 90, "클릭수": 30, "결제액": 400, "에러수": 2, "스크롤깊이": 150},
    "3. 서버 장애 발생 (에러 폭주)": {"체류시간": 20, "클릭수": 5, "결제액": 10, "에러수": 25, "스크롤깊이": 30}
}

# ==========================================
# 세션 상태(Session State) 초기화
# ==========================================
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "1. 평범한 평일 (기본)"

# 🎯 [핵심 수정] 검색창과 그래프 클릭을 완벽하게 동기화할 전용 세션 키 생성
if 'selected_user_id' not in st.session_state:
    st.session_state.selected_user_id = "선택 안 함"

for key, val in SCENARIO_DEFAULTS[st.session_state.current_scenario].items():
    if f"cfg_{key}" not in st.session_state:
        st.session_state[f"cfg_{key}"] = val

def 세션_기준값_업데이트(유저_행_데이터):
    st.session_state["cfg_체류시간"] = int(round(유저_행_데이터["체류시간"]))
    st.session_state["cfg_클릭수"] = int(round(유저_행_데이터["클릭수"]))
    st.session_state["cfg_결제액"] = int(round(유저_행_데이터["결제액"]))
    st.session_state["cfg_에러수"] = int(round(유저_행_데이터["에러수"]))
    st.session_state["cfg_스크롤깊이"] = int(round(유저_행_데이터["스크롤깊이"]))

# ==========================================
# 사이드바 (시스템 설정)
# ==========================================
st.sidebar.header("⚙️ 시스템 설정")

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
st.sidebar.markdown("### ✍️ 정상치 기준값(평균) 직접 입력")
if st.sidebar.button("↩️ 현재 시나리오 기본값으로 초기화", use_container_width=True):
    for key, val in SCENARIO_DEFAULTS[시나리오].items():
        st.session_state[f"cfg_{key}"] = val
    st.rerun()

정상_체류 = st.sidebar.number_input("정상 체류시간 평균", min_value=0, max_value=10000, step=1, key="cfg_체류시간")
정상_클릭 = st.sidebar.number_input("정상 클릭수 평균", min_value=0, max_value=5000, step=1, key="cfg_클릭수")
정상_결제 = st.sidebar.number_input("정상 결제액 평균", min_value=0, max_value=100000, step=10, key="cfg_결제액")
정상_에러 = st.sidebar.number_input("정상 에러수 평균", min_value=0, max_value=1000, step=1, key="cfg_에러수")
정상_스크롤 = st.sidebar.number_input("정상 스크롤깊이 평균", min_value=0, max_value=10000, step=1, key="cfg_스크롤깊이")

st.sidebar.markdown("---")
정상_샘플수 = st.sidebar.slider("일반 유저 데이터 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 데이터 수", 10, 500, 50, 10)
위험_비율 = st.sidebar.slider("🔴 위험(차단) 판정 비율", min_value=1, max_value=20, value=6, step=1, format="%d%%")
주의_비율 = st.sidebar.slider("🟡 주의(관찰) 판정 비율", min_value=5, max_value=30, value=10, step=1, format="%d%%")

# ==========================================
# 데이터 생성 및 위험도 평가 로직
# ==========================================
np.random.seed(st.session_state.random_seed)
시나리오_원래값 = SCENARIO_DEFAULTS[시나리오]
진짜_기본_기준점 = [시나리오_원래값['체류시간'], 시나리오_원래값['클릭수'], 시나리오_원래값['결제액'], 시나리오_원래값['에러수'], 시나리오_원래값['스크롤깊이']]

정상_시그마 = np.array([10, 2, 20, 0.5, 10])
정상_데이터 = np.random.normal(loc=진짜_기본_기준점, scale=정상_시그마, size=(정상_샘플수, 5))
이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5))

전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
df = pd.DataFrame(전체_데이터, columns=기본_특성)
df[기본_특성] = df[기본_특성].clip(lower=0) 
df.insert(0, '유저 번호', [f"USR-{i+1:04d}" for i in range(len(df))])

설정_기준점 = np.array([정상_체류, 정상_클릭, 정상_결제, 정상_에러, 정상_스크롤])
z_scores = np.abs((df[기본_특성].values - 설정_기준점) / 정상_시그마)
df['위험도 점수'] = np.round(np.clip((np.max(z_scores, axis=1) / 5.5) * 100, 0, 100), 1)

위험_문턱 = df['위험도 점수'].quantile(1.0 - (위험_비율 / 100.0))
주의_문턱 = df['위험도 점수'].quantile(1.0 - ((위험_비율 + 주의_비율) / 100.0))

def classify_status(row):
    if row['위험도 점수'] >= 위험_문턱 or row['위험도 점수'] >= 80.0: return '🔴 위험 (차단 대상)'
    elif row['위험도 점수'] >= 주의_문턱 or row['위험도 점수'] >= 50.0: return '🟡 주의 (관찰 요망)'
    else: return '🔵 안전 (정상 패턴)'

df['상태'] = df.apply(classify_status, axis=1)
df['마커크기'] = df['상태'].map({'🔵 안전 (정상 패턴)': 5, '🟡 주의 (관찰 요망)': 10, '🔴 위험 (차단 대상)': 18})

설정_시리즈 = pd.Series(설정_기준점, index=기본_특성)
df['주요원인_특성'] = np.abs((df[기본_특성] - 설정_시리즈) / 정상_시그마).idxmax(axis=1)

def format_hover_text(row, col, 기준값):
    편차 = row[col] - 기준값
    기본문구 = f"{row[col]:.1f} (기준대비 {'+' if 편차 > 0 else ''}{편차:.1f})"
    if row['상태'] != '🔵 안전 (정상 패턴)' and row['주요원인_특성'] == col:
        색상 = "#D32F2F" if '🔴 위험' in row['상태'] else "#E65100"
        return f'<b><span style="color:{색상};">{기본문구} ◀ 원인</span></b>'
    return 기본문구

for 특성, 기준 in zip(기본_특성, 설정_기준점):
    df[f'{특성}_표시'] = df.apply(lambda row: format_hover_text(row, 특성, 기준), axis=1)

pca = PCA(n_components=3)
잠재공간 = pca.fit_transform(df[기본_특성])
df['잠재축 X'], df['잠재축 Y'], df['잠재축 Z'] = 잠재공간[:, 0], 잠재공간[:, 1], 잠재공간[:, 2]

# ==========================================
# 화면 분할 및 그래프 시각화
# ==========================================
좌측_화면, 우측_화면 = st.columns([3, 1.2]) 

with 좌측_화면:
    st.markdown("⚠️ **안내:** 유저 점을 클릭하여 프로필을 보시려면 반드시 **[2D 평면 뷰]**를 선택해주세요. (3D는 클릭 불가)")
    뷰_모드 = st.radio("모드 선택", ["🌐 3D 뷰 (전체 분포 확인)", "🗺️ 2D 평면 뷰 (클릭하여 유저 프로필 연동)"], horizontal=True, label_visibility="collapsed")
    
    호버_템플릿 = (
        "<b>[%{customdata[0]}] %{customdata[1]}</b><br>"
        "위험도 점수: %{customdata[2]} / 100 점<br>"
        "-----------------------------------<br>"
        "체류시간 : %{customdata[3]}<br>"
        "클릭수   : %{customdata[4]}<br>"
        "결제액   : %{customdata[5]}<br>"
        "에러수   : %{customdata[6]}<br>"
        "스크롤   : %{customdata[7]}<br>"
        "<extra></extra>"
    )
    
    선택_결과 = None
    
    if "2D" in 뷰_모드:
        fig = px.scatter(
            df, x='잠재축 X', y='잠재축 Y', color='상태',
            color_discrete_map={'🔵 안전 (정상 패턴)': 'rgba(30, 136, 229, 0.4)', '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.9)', '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'},
            size='마커크기', size_max=18,
            custom_data=['유저 번호', '상태', '위험도 점수', '체류시간_표시', '클릭수_표시', '결제액_표시', '에러수_표시', '스크롤깊이_표시'],
            template='plotly_dark'
        )
        fig.update_traces(hovertemplate=호버_템플릿)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=13, font_color="black"), margin=dict(l=0, r=0, b=0, t=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        # 2D 클릭 이벤트 리스너 활성화
        선택_결과 = st.plotly_chart(fig, use_container_width=True, height=650, on_select="rerun")
    else:
        fig = px.scatter_3d(
            df, x='잠재축 X', y='잠재축 Y', z='잠재축 Z', color='상태',
            color_discrete_map={'🔵 안전 (정상 패턴)': 'rgba(30, 136, 229, 0.15)', '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.8)', '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'},
            size='마커크기', size_max=15,
            custom_data=['유저 번호', '상태', '위험도 점수', '체류시간_표시', '클릭수_표시', '결제액_표시', '에러수_표시', '스크롤깊이_표시'],
            template='plotly_dark'
        )
        fig.update_traces(hovertemplate=호버_템플릿)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=13, font_color="black"), margin=dict(l=0, r=0, b=0, t=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True, height=650)

# ==========================================
# 🎯 [핵심 수정] 2D 그래프 클릭 이벤트 -> 우측 정보창 즉시 동기화 로직
# ==========================================
if 선택_결과 and "selection" in 선택_결과:
    포인트들 = 선택_결과["selection"].get("points", [])
    if 포인트들:
        클릭된_유저_아이디 = 포인트들[0]["customdata"][0]
        # 클릭한 유저가 기존 세션 값과 다르면 갱신 후 즉시 새로고침(rerun)
        if st.session_state.selected_user_id != 클릭된_유저_아이디:
            st.session_state.selected_user_id = 클릭된_유저_아이디
            st.rerun()

유저_옵션_리스트 = ["선택 안 함"] + list(df['유저 번호'].values)

with 우측_화면:
    st.subheader("🔍 개별 유저 상세 정보")
    
    # 세션 키(key)를 묶어주어 그래프를 클릭할 때마다 이 selectbox가 자동으로 바뀜
    선택된_유저 = st.selectbox(
        "2D 뷰에서 점을 클릭하거나 직접 검색하세요 👆",
        options=유저_옵션_리스트,
        key="selected_user_id"
    )
    
    if 선택된_유저 != "선택 안 함":
        유저_데이터 = df[df['유저 번호'] == 선택된_유저].iloc[0]
        
        with st.container(border=True):
            st.markdown(f"### 👤 {선택된_유저} 프로필")
            st.markdown(f"**현재 상태:** {유저_데이터['상태']}")
            st.markdown(f"**위험도 점수:** `{유저_데이터['위험도 점수']}` / 100 점")
            
            if 유저_데이터['상태'] != '🔵 안전 (정상 패턴)':
                st.markdown(f"⚠️ **주요 특이 원인:** <span style='color:#FF5722; font-weight:bold;'>{유저_데이터['주요원인_특성']}</span>", unsafe_allow_html=True)
            
            st.button(
                "🎯 이 유저의 지표를 정상 기준으로 설정", 
                use_container_width=True,
                on_click=세션_기준값_업데이트,
                args=(유저_데이터,)
            )

            st.markdown("**📋 상세 지표 현황**")
            지표_데이터 = {}
            for 특성, 기준 in zip(기본_특성, 설정_기준점):
                값 = 유저_데이터[특성]
                편차 = 값 - 기준
                지표_데이터[특성] = f"{값:.1f} (정상대비 {'+' if 편차 > 0 else ''}{편차:.1f})"
            st.json(지표_데이터)
            
    st.markdown("---")
    st.write("**⚠️ 실시간 고위험 유저 (Top 5)**")
    보여줄_컬럼 = ['유저 번호', '상태', '위험도 점수'] + 기본_특성
    st.dataframe(df[df['상태'] == '🔴 위험 (차단 대상)'][보여줄_컬럼].sort_values(by='위험도 점수', ascending=False).head(5), hide_index=True)
