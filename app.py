import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="이상 패턴 탐지 대시보드", layout="wide")

st.title("🚀 사용자 행동 잠재공간 및 이상 패턴 탐지 시스템")

# ==========================================
# 1. 초기 설정 및 상태 관리
# ==========================================
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

# ==========================================
# 2. 사이드바 (시스템 설정)
# ==========================================
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

st.sidebar.markdown("---")
정상_샘플수 = st.sidebar.slider("일반 유저 수", 100, 2000, 800, 100)
이상_샘플수 = st.sidebar.slider("이상 유저 수", 10, 500, 50, 10)
위험_비율 = st.sidebar.slider("🔴 위험 판정 비율", 1, 20, 6, 1, format="%d%%")
주의_비율 = st.sidebar.slider("🟡 주의 판정 비율", 5, 30, 10, 1, format="%d%%")

# ==========================================
# 3. 데이터 생성 및 PCA 캐싱 (🔥 렉 방지의 핵심)
# ==========================================
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

    # 가장 무거운 PCA 연산을 최초 1회만 수행하고 저장
    pca = PCA(n_components=3)
    잠재공간 = pca.fit_transform(base_df[기본_특성])
    base_df['잠재축 X'], base_df['잠재축 Y'], base_df['잠재축 Z'] = 잠재공간[:, 0], 잠재공간[:, 1], 잠재공간[:, 2]
    
    s
