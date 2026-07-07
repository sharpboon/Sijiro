import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
# [변경] streamlit-plotly-events 제거. Streamlit 1.35+ 내장 on_select 기능 사용.
# (plotly_events는 커스텀 컴포넌트라 Community Cloud에서 로딩 실패/좌표 오차 이슈가 잦음)

st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("🚀 사용자 행동 잠재공간 및 이상 패턴 탐지 시스템")

# 1. 초기 설정 및 상태 관리
SCENARIO_DEFAULTS = {
    "1. 평범한 평일": {"체류시간": 50, "클릭수": 10, "결제액": 100, "에러수": 1, "스크롤깊이": 80},
    "2. 주말 대규모 이벤트": {"체류시간": 90, "클릭수": 30, "결제액": 400, "에러수": 2, "스크롤깊이": 150},
    "3. 서버 장애 발생": {"체류시간": 20, "클릭수": 5, "결제액": 10, "에러수": 25, "스크롤깊이": 30}
}

if 'random_seed' not in st.session_state: st.session_state.random_seed = 42
if 'current_scenario' not in st.session_state: st.session_state.current_scenario = "1. 평범한 평일"
if 'selected_user_id' not in st.session_state: st.session_state.selected_user_id = "선택 안 함"
if 'last_settings' not in st.session_state: st.session_state.last_settings = {}

for key, val in SCENARIO_DEFAULTS[st.session_state.current_scenario].items():
    if f"cfg_{key}" not in st.session_state:
        st.session_state[f"cfg_{key}"] = val

# 2. 사이드바 (시스템 설정)
st.sidebar.header("⚙️ 시스템 설정")
if st.sidebar.button("🔄 새로운 무작위 유저 생성", use_container_width=True):
    st.session_state.random_seed = np.random.randint(0, 100000)

시나리오 = st.sidebar.selectbox("시나리오 환경 선택", list(SCENARIO_DEFAULTS.keys()), index=list(SCENARIO_DEFAULTS.keys()).index(st.session_state.current_scenario))

if 시나리오 != st.session_state.current_scenario:
    st.session_state.current_scenario = 시나리오
    for key, val in SCENARIO_DEFAULTS[시나리오].items(): st.session_state[f"cfg_{key}"] = val
    st.rerun()

st.sidebar.markdown("---")
정상_체류 = st.sidebar.number_input("정상 체류시간", min_value=0, max_value=10000, step=1, key="cfg_체류시간")
정상_클릭 = st.sidebar.number_input("정상 클릭수", min_value=0, max_value=5000, step=1, key="cfg_클릭수")
정상_결제 = st.sidebar.number_input("정상 결제액", min_value=0, max_value=100000, step=10, key="cfg_결제액")
정상_에러 = st.sidebar.number_input("정상 에러수", min_value=0, max_value=1000, step=1, key="cfg_에러수")
정상_스크롤 = st.sidebar.number_input("정상 스크롤깊이", min_value=0, max_value=10000, step=1, key="cfg_스크롤깊이")

정상_샘플수 = st.sidebar.slider("일반 유저 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 수", 10, 500, 50, 10)
위험_비율 = st.sidebar.slider("🔴 위험 판정 비율", 1, 20, 6, 1, format="%d%%")
주의_비율 = st.sidebar.slider("🟡 주의 판정 비율", 5, 30, 10, 1, format="%d%%")

# 3. 데이터 생성 및 PCA 캐싱 (기존 방식 유지: 클릭 시 재계산 안 됨 - 이미 올바르게 구현되어 있었음)
현재_설정 = {"seed": st.session_state.random_seed, "scenario": st.session_state.current_scenario, "n_norm": 정상_샘플수, "n_anom": 이상_샘플수}

if st.session_state.last_settings != 현재_설정 or 'cached_base_df' not in st.session_state:
    np.random.seed(st.session_state.random_seed)
    진짜_기본_기준점 = list(SCENARIO_DEFAULTS[시나리오].values())
    정상_시그마 = np.array([10, 2, 20, 0.5, 10])

    정상_데이터 = np.random.normal(loc=진짜_기본_기준점, scale=정상_시그마, size=(정상_샘플수, 5))
    이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5))

    전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
    기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
    base_df = pd.DataFrame(전체_데이터, columns=기본_특성).clip(lower=0)
    base_df.insert(0, '유저 번호', [f"USR-{i+1:04d}" for i in range(len(base_df))])

    pca = PCA(n_components=3)
    잠재공간 = pca.fit_transform(base_df[기본_특성])
    base_df['잠재축 X'], base_df['잠재축 Y'], base_df['잠재축 Z'] = 잠재공간[:, 0], 잠재공간[:, 1], 잠재공간[:, 2]

    st.session_state.cached_base_df = base_df
    st.session_state.last_settings = 현재_설정

df = st.session_state.cached_base_df.copy()
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
정상_시그마 = np.array([10, 2, 20, 0.5, 10])

# 4. 실시간 위험도 평가
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
df['마커크기'] = df['상태'].map({'🔵 안전 (정상 패턴)': 5, '🟡 주의 (관찰 요망)': 10, '🔴 위험 (차단 대상)': 20})

설정_시리즈 = pd.Series(설정_기준점, index=기본_특성)
df['주요원인_특성'] = np.abs((df[기본_특성] - 설정_시리즈) / 정상_시그마).idxmax(axis=1)

# 5. 화면 분할 및 3D 그래프 생성
좌측_화면, 우측_화면 = st.columns([3, 1.2])

with 좌측_화면:
    호버_템플릿 = "<b>[%{customdata[0]}] %{customdata[1]}</b><br>위험도 점수: %{customdata[2]} / 100 점<br><extra></extra>"

    fig = px.scatter_3d(
        df, x='잠재축 X', y='잠재축 Y', z='잠재축 Z', color='상태',
        color_discrete_map={'🔵 안전 (정상 패턴)': 'rgba(30, 136, 229, 0.25)', '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.9)', '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'},
        size='마커크기', size_max=20, custom_data=['유저 번호', '상태', '위험도 점수'], template='plotly_dark'
    )
    fig.update_traces(hovertemplate=호버_템플릿)
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='#333333', title='활동성 (X)'),
            yaxis=dict(showbackground=False, gridcolor='#333333', title='결제/에러 (Y)'),
            zaxis=dict(showbackground=False, gridcolor='#333333', title='변동성 (Z)'),
            hovermode='closest'
        )
        # [변경] xaxis_clickmode='event' 같은 존재하지 않는 3D scene 속성 절대 사용 안 함
    )

    # [핵심 변경] plotly_events(fig, click_event=True) 대신 st.plotly_chart의 내장 on_select 사용.
    # Streamlit 1.35+ 필요 (requirements.txt에 streamlit>=1.35 명시 권장).
    선택_이벤트 = st.plotly_chart(
        fig,
        use_container_width=True,
        key="3d_scatter_chart",
        on_select="rerun",
        selection_mode="points",
    )

# 6. 클릭된 유저 판별 로직
# [핵심 변경] 좌표(x,y,z) 유클리디안 거리 근사치 매칭을 완전히 제거.
# custom_data로 이미 넣어둔 '유저 번호'를 이벤트에서 직접 꺼내 쓴다.
# -> 색상(color='상태')별로 trace가 여러 개로 나뉘어도, point_index가 아니라
#    customdata를 쓰므로 어떤 trace의 점이든 정확히 유저 번호가 매칭된다.
클릭된_포인트_목록 = 선택_이벤트["selection"]["points"] if 선택_이벤트 else []

if 클릭된_포인트_목록:
    클릭된_유저_아이디 = 클릭된_포인트_목록[0]["customdata"][0]

    if 클릭된_유저_아이디 in df['유저 번호'].values and st.session_state.selected_user_id != 클릭된_유저_아이디:
        st.session_state.selected_user_id = 클릭된_유저_아이디
        st.rerun()

# 7. 우측 상세 프로필
유저_옵션_리스트 = ["선택 안 함"] + list(df['유저 번호'].values)

def 기준값_동기화(유저_데이터):
    st.session_state["cfg_체류시간"] = int(round(유저_데이터["체류시간"]))
    st.session_state["cfg_클릭수"] = int(round(유저_데이터["클릭수"]))
    st.session_state["cfg_결제액"] = int(round(유저_데이터["결제액"]))
    st.session_state["cfg_에러수"] = int(round(유저_데이터["에러수"]))
    st.session_state["cfg_스크롤깊이"] = int(round(유저_데이터["스크롤깊이"]))

with 우측_화면:
    선택된_유저 = st.selectbox("3D 그래프에서 점을 클릭하세요 👆", options=유저_옵션_리스트, key="selected_user_id")

    if 선택된_유저 != "선택 안 함":
        유저_데이터 = df[df['유저 번호'] == 선택된_유저].iloc[0]
        with st.container(border=True):
            st.markdown(f"### 👤 {선택된_유저}")
            st.markdown(f"**현재 상태:** {유저_데이터['상태']}")
            st.markdown(f"**위험도 점수:** `{유저_데이터['위험도 점수']}` / 100 점")
            st.button("🎯 이 유저를 정상 기준으로 설정", use_container_width=True, on_click=기준값_동기화, args=(유저_데이터,))
