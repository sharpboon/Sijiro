import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

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
기본_특성 = ['체류시간', '클릭수', '결제액', '에러수', '스크롤깊이']
정상_시그마 = np.array([10, 2, 20, 0.5, 10])

# ==========================================
# 데이터 생성 + PCA를 캐싱
# ==========================================
@st.cache_data(show_spinner=False)
def 데이터_생성_및_PCA(시드, 시나리오_키, 정상_샘플수, 이상_샘플수):
    np.random.seed(시드)
    시나리오_원래값 = SCENARIO_DEFAULTS[시나리오_키]
    진짜_기본_기준점 = [시나리오_원래값[k] for k in 기본_특성]

    # 샘플 수가 0일 경우를 대비해 빈 배열도 안전하게 생성되도록 설정
    정상_데이터 = np.random.normal(loc=진짜_기본_기준점, scale=정상_시그마, size=(정상_샘플수, 5)) if 정상_샘플수 > 0 else np.empty((0, 5))
    이상_데이터 = np.random.uniform(low=[10, 50, 0, 10, 10], high=[100, 200, 500, 50, 100], size=(이상_샘플수, 5)) if 이상_샘플수 > 0 else np.empty((0, 5))

    전체_데이터 = np.vstack([정상_데이터, 이상_데이터])
    base_df = pd.DataFrame(전체_데이터, columns=기본_특성).clip(lower=0)
    base_df.insert(0, '유저 번호', [f"USR-{i+1:04d}" for i in range(len(base_df))])

    pca = PCA(n_components=3)
    잠재공간 = pca.fit_transform(base_df[기본_특성])
    base_df['잠재축 X'], base_df['잠재축 Y'], base_df['잠재축 Z'] = 잠재공간[:, 0], 잠재공간[:, 1], 잠재공간[:, 2]
    return base_df

# ==========================================
# 위젯 콜백 함수
# ==========================================
def 세션_기준값_업데이트(유저_행_데이터):
    st.session_state["cfg_체류시간"] = int(round(유저_행_데이터["체류시간"]))
    st.session_state["cfg_클릭수"] = int(round(유저_행_데이터["클릭수"]))
    st.session_state["cfg_결제액"] = int(round(유저_행_데이터["결제액"]))
    st.session_state["cfg_에러수"] = int(round(유저_행_데이터["에러수"]))
    st.session_state["cfg_스크롤깊이"] = int(round(유저_행_데이터["스크롤깊이"]))

# ==========================================
# 세션 상태 초기화
# ==========================================
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "1. 평범한 평일 (기본)"

for key, val in SCENARIO_DEFAULTS[st.session_state.current_scenario].items():
    if f"cfg_{key}" not in st.session_state:
        st.session_state[f"cfg_{key}"] = val

# ==========================================
# 사이드바
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
st.sidebar.markdown("### 📊 모니터링 데이터 규모 (직접 입력)")
# [수정] 최솟값을 0으로 내리고, step을 1로 설정하여 1명씩 증감 및 0 입력이 가능하도록 변경
정상_샘플수 = st.sidebar.number_input("일반 유저 데이터 수", min_value=0, max_value=100000, value=800, step=1)
이상_샘플수 = st.sidebar.number_input("이상 유저 데이터 수", min_value=0, max_value=50000, value=50, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 판정 민감도 (상세 조건 설정)")

위험_콜1, 위험_콜2 = st.sidebar.columns([3, 2])
위험_점수 = 위험_콜1.number_input("🔴 차단 점수", min_value=1, max_value=100, value=80, step=1)
위험_연산자 = 위험_콜2.selectbox("조건", ["이상 (≥)", "초과 (>)"], key="danger_op")

주의_콜1, 주의_콜2 = st.sidebar.columns([3, 2])
주의_점수 = 주의_콜1.number_input("🟡 주의 점수", min_value=1, max_value=100, value=50, step=1)
주의_연산자 = 주의_콜2.selectbox("조건", ["이상 (≥)", "초과 (>)"], key="caution_op")

if 주의_점수 >= 위험_점수:
    st.sidebar.error("⚠️ 오류: '🔴 차단' 점수가 '🟡 주의' 점수보다 높아야 합니다.")

# ==========================================
# [추가] 데이터 부족 예외 처리 가드레일 (UI 터짐 방지)
# ==========================================
if 정상_샘플수 + 이상_샘플수 < 3:
    st.error("🚨 **데이터 부족 예외 발생:** 3차원 공간(PCA)을 형성하기 위해서는 전체 유저 수가 **최소 3명 이상**이어야 합니다. 사이드바에서 데이터 수를 늘려주세요.")
    st.stop()

# ==========================================
# 캐싱된 원본 데이터 로드
# ==========================================
base_df = 데이터_생성_및_PCA(st.session_state.random_seed, 시나리오, 정상_샘플수, 이상_샘플수)
df = base_df.copy()

# ==========================================
# 실시간 위험도 평가 및 논리 연산
# ==========================================
설정_기준점 = np.array([정상_체류, 정상_클릭, 정상_결제, 정상_에러, 정상_스크롤])
z_scores = np.abs((df[기본_특성].values - 설정_기준점) / 정상_시그마)
최대_이탈도 = np.max(z_scores, axis=1)
df['위험도 점수'] = np.round(np.clip((최대_이탈도 / 5.5) * 100, 0, 100), 1)

if 위험_연산자 == "이상 (≥)":
    조건_위험 = df['위험도 점수'] >= 위험_점수
else:
    조건_위험 = df['위험도 점수'] > 위험_점수

if 주의_연산자 == "이상 (≥)":
    조건_주의 = df['위험도 점수'] >= 주의_점수
else:
    조건_주의 = df['위험도 점수'] > 주의_점수

df['상태'] = np.select([조건_위험, 조건_주의], ['🔴 위험 (차단 대상)', '🟡 주의 (관찰 요망)'], default='🔵 안전 (정상 패턴)')
df['마커크기'] = df['상태'].map({'🔵 안전 (정상 패턴)': 2, '🟡 주의 (관찰 요망)': 6, '🔴 위험 (차단 대상)': 15})

설정_시리즈 = pd.Series(설정_기준점, index=기본_특성)
표준점수 = np.abs((df[기본_특성] - 설정_시리즈) / 정상_시그마)
df['주요원인_특성'] = 표준점수.idxmax(axis=1)

안전_마스크 = (df['상태'] == '🔵 안전 (정상 패턴)').values
위험_마스크 = df['상태'].str.contains('위험').values
for 특성, 기준 in zip(기본_특성, 설정_기준점):
    값 = df[특성].values
    편차 = 값 - 기준
    편차표시 = np.where(편차 > 0, [f"+{d:.1f}" for d in 편차], [f"{d:.1f}" for d in 편차])
    기본문구 = np.array([f"{v:.1f} (기준대비 {d})" for v, d in zip(값, 편차표시)])
    원인_마스크 = (~안전_마스크) & (df['주요원인_특성'].values == 특성)
    강조문구 = np.where(
        위험_마스크,
        [f'<b><span style="color:#D32F2F;">{t} ◀ 원인</span></b>' for t in 기본문구],
        [f'<b><span style="color:#E65100;">{t} ◀ 원인</span></b>' for t in 기본문구]
    )
    df[f'{특성}_표시'] = np.where(원인_마스크, 강조문구, 기본문구)

# ==========================================
# 안내 가이드
# ==========================================
안전_조건텍스트 = f"{주의_점수}점 {'미만' if 주의_연산자 == '이상 (≥)' else '이하'}"
주의_하한텍스트 = f"{주의_점수}점 {'이상' if 주의_연산자 == '이상 (≥)' else '초과'}"
주의_상한텍스트 = f"{위험_점수}점 {'미만' if 위험_연산자 == '이상 (≥)' else '이하'}"
위험_조건텍스트 = f"{위험_점수}점 {'이상' if 위험_연산자 == '이상 (≥)' else '초과'}"

with st.expander("💡 실시간 등급 판정 및 데이터별 이상 원인 분석 기준 안내", expanded=False):
    공지_좌, 공지_우 = st.columns(2)
    with 공지_좌:
        st.markdown(f"""
        #### 📌 3단계 탐지 등급 산정 기준
        * **🔵 안전**: 위험 점수 **{안전_조건텍스트}**
        * **🟡 주의**: 위험 점수 **{주의_하한텍스트} ~ {주의_상한텍스트}**
        * **🔴 위험**: 위험 점수 **{위험_조건텍스트}**
        """)
    with 공지_우:
        st.markdown("""
        #### 🔍 원인(`◀ 원인`) 도출 방식
        * 유저 고유 수치는 고정, 사이드바 기준값 변경 시 편차만 실시간 재연산
        * 기준 평균치에서 통계적으로 가장 멀리 이탈한 지표에 표시
        """)

# ==========================================
# 메인 3D 그래프
# ==========================================
좌측_화면, 우측_화면 = st.columns([3, 1.2])

with 좌측_화면:
    fig = px.scatter_3d(
        df, x='잠재축 X', y='잠재축 Y', z='잠재축 Z',
        color='상태',
        color_discrete_map={
            '🔵 안전 (정상 패턴)': 'rgba(0, 102, 255, 1.0)',
            '🟡 주의 (관찰 요망)': 'rgba(255, 193, 7, 0.85)',
            '🔴 위험 (차단 대상)': 'rgba(255, 30, 30, 1.0)'
        },
        size='마커크기', size_max=15,
        custom_data=['유저 번호', '상태', '위험도 점수', '체류시간_표시', '클릭수_표시', '결제액_표시', '에러수_표시', '스크롤깊이_표시'],
        template='plotly_dark'
    )
    fig.update_traces(
        hovertemplate=(
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
    )
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=13, font_color="black", bordercolor="black"),
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='주요 활동성 (X)'),
            yaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='결제/에러 (Y)'),
            zaxis=dict(showbackground=False, gridcolor='#333333', zerolinecolor='gray', title='행동 변동성 (Z)')
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 우측 화면: 검색창
# ==========================================
with 우측_화면:
    st.markdown("#### 🔍 유저 상세 검색")
    유저_옵션 = ["선택 안 함"] + list(df['유저 번호'].values)
    if st.session_state.get("selected_user_search") not in 유저_옵션:
        st.session_state["selected_user_search"] = "선택 안 함"

    선택된_유저 = st.selectbox(
        "유저 번호를 선택하거나 타이핑하여 검색하세요.",
        options=유저_옵션,
        key="selected_user_search"
    )
    st.caption("⬇️ 통계 · TOP10 · 상세 분석은 그래프 아래에 표시됩니다.")

# ==========================================
# 유저 상세 분석
# ==========================================
if 선택된_유저 != "선택 안 함":
    st.divider()
    st.subheader(f"👤 {선택된_유저} 상세 분석")

    유저_데이터 = df[df['유저 번호'] == 선택된_유저].iloc[0]

    상세_좌, 상세_우 = st.columns([1.2, 1])

    with 상세_좌:
        st.markdown(f"**현재 상태:** {유저_데이터['상태']}")
        st.markdown(f"**위험도 점수:** `{유저_데이터['위험도 점수']}` / 100 점")
        if 유저_데이터['상태'] != '🔵 안전 (정상 패턴)':
            st.markdown(f"⚠️ **주요 특이 원인:** <span style='color:#FF5722; font-weight:bold;'>{유저_데이터['주요원인_특성']}</span>", unsafe_allow_html=True)

        st.button(
            "🎯 이 유저의 지표를 정상 기준으로 설정",
            on_click=세션_기준값_업데이트,
            args=(유저_데이터,)
        )

        st.markdown("**📋 상세 지표 현황**")
        지표_행 = []
        for 특성, 기준, sigma in zip(기본_특성, 설정_기준점, 정상_시그마):
            값 = 유저_데이터[특성]
            편차 = 값 - 기준
            z = abs(편차 / sigma)
            지표_행.append({
                "지표": 특성,
                "값": f"{값:.1f}",
                "기준대비": f"{'+' if 편차 > 0 else ''}{편차:.1f}",
                "Z-score": round(z, 2),
                "판정": "🔴 심각" if z >= 3 else ("🟡 주의" if z >= 2 else "🟢 정상")
            })
        지표_표 = pd.DataFrame(지표_행)
        st.dataframe(지표_표, hide_index=True, use_container_width=True)

    with 상세_우:
        기여도 = pd.Series({r["지표"]: r["Z-score"] for r in 지표_행})
        if 기여도.sum() > 0:
            비율 = (기여도 / 기여도.sum() * 100).sort_values(ascending=False)
            fig2 = px.bar(x=비율.values, y=비율.index, orientation='h', labels={'x': '기여도(%)', 'y': '변수'})
            fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

            top = 비율.index[0]
            txt = f"가장 큰 이상 원인은 **{top}**이며, "
            if len(비율[비율 > 15]) > 1:
                txt += f"{', '.join(list(비율[비율 > 15].index)[1:])}도 함께 위험도 증가에 영향을 주었습니다."
            else:
                txt += "다른 변수들은 상대적으로 정상 범위에 가깝습니다."
            st.info(txt)

# ==========================================
# 통계 · TOP10
# ==========================================
st.divider()
st.markdown("#### 📊 탐지 통계 · TOP 10")

안전_수 = int((df['상태'] == '🔵 안전 (정상 패턴)').sum())
주의_수 = int((df['상태'] == '🟡 주의 (관찰 요망)').sum())
위험_수 = int((df['상태'] == '🔴 위험 (차단 대상)').sum())

통계1, 통계2, 통계3, 통계4 = st.columns(4)
통계1.metric(label="전체 모니터링 대상", value=f"{len(df)} 명")
통계2.metric(label="🔵 안전 유저", value=f"{안전_수} 명")
통계3.metric(label="🟡 주의 행동 감지", value=f"{주의_수} 건")
통계4.metric(label="🔴 위험 행동 감지", value=f"{위험_수} 건")

st.write("**⚠️ 실시간 고위험 유저 (Top 10)**")
보여줄_컬럼 = ['유저 번호', '상태', '위험도 점수'] + 기본_특성
st.dataframe(
    df[df['상태'] == '🔴 위험 (차단 대상)'][보여줄_컬럼]
    .sort_values(by='위험도 점수', ascending=False)
    .head(10),
    hide_index=True,
    use_container_width=True
)
