import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="극지식물 생육 상관관계 분석 대시보드", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

st.sidebar.header("생육 측정 항목")
measure_select = st.sidebar.selectbox(
    "생육 측정 항목",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"],
    key="measure_select"
)

def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

@st.cache_data
def load_data(file):
    return std_cols(read_csv(file))

# ── Data loading ───────────────────────────────────────────────────────────
env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = ""

if growth_file is not None:
    with st.spinner("생육 데이터를 읽고 있습니다..."):
        growth_df = load_data(growth_file)
    school_name = growth_file.name.split(".")[0]
elif env_file is not None:
    with st.spinner("환경 데이터를 읽고 있습니다..."):
        env_df = load_data(env_file)
    school_name = env_file.name.split(".")[0]

st.sidebar.markdown("---")
if not growth_df.empty:
    st.sidebar.success(f"✅ 생육 데이터 로드 완료: {school_name} ({len(growth_df)}개 개체)")
else:
    st.sidebar.info("생육 결과 CSV를 업로드해 주세요.")
if not env_df.empty:
    st.sidebar.success(f"✅ 환경 데이터 로드 완료 ({len(env_df)}개 시점)")
else:
    st.sidebar.info("환경 데이터 CSV를 업로드해 주세요.")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["실험 소개", "생육 결과", "실생활 대입"])

# ── Tab 1: 실험 소개 ──────────────────────────────────────────────────────
with tab1:
    try:
        st.title("지상부 요소로 지하부 상태 예측하기")
        st.subheader("극지식물 생육 상관관계 분석 대시보드")
        if school_name:
            st.write(f"학교: {school_name}")

        st.info("**연구 질문**\n\n\"지상부의 어떤 요소를 보고 지하부의 상태를 예측할 수 있을까?\"")

        st.success("**연구 가설**\n\n1. 잎 수가 많고 지상부 길이가 긴 개체일수록 지하부도 길고 무거울 것이다.\n2. 지상부 요소(잎 수, 지상부 길이)는 지하부 상태와 양의 상관관계를 보일 것이다.")

        st.warning("**목적**\n\n• 지상부 생육 요소만으로 지하부 상태를 추정할 수 있는지 검증\n• 스마트팜에서 지하부 직접 측정이 어려운 경우 지상부 측정값 활용 가능성 탐색\n• 극지식물 개체 간 생육 차이 패턴 파악")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if not growth_df.empty and "개체번호" in growth_df.columns:
                st.metric("전체 개체 수", len(growth_df))
            else:
                st.metric("전체 개체 수", "데이터 없음")
        with col2:
            if not growth_df.empty and "생중량" in growth_df.columns:
                avg_w = growth_df["생중량"].dropna()
                if len(avg_w) > 0:
                    st.metric("평균 생중량", f"{avg_w.mean():.2f} g")
                else:
                    st.metric("평균 생중량", "데이터 없음")
            else:
                st.metric("평균 생중량", "데이터 없음")
        with col3:
            if not growth_df.empty and "지상부 길이" in growth_df.columns:
                avg_s = growth_df["지상부 길이"].dropna()
                if len(avg_s) > 0:
                    st.metric("평균 지상부 길이", f"{avg_s.mean():.1f} mm")
                else:
                    st.metric("평균 지상부 길이", "데이터 없음")
            else:
                st.metric("평균 지상부 길이", "데이터 없음")
        with col4:
            if not growth_df.empty and "지하부 길이" in growth_df.columns:
                avg_u = growth_df["지하부 길이"].dropna()
                if len(avg_u) > 0:
                    st.metric("평균 지하부 길이", f"{avg_u.mean():.1f} mm")
                else:
                    st.metric("평균 지하부 길이", "데이터 없음")
            else:
                st.metric("평균 지하부 길이", "데이터 없음")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 2: 생육 결과 ──────────────────────────────────────────────────────
with tab2:
    try:
        st.subheader("생육 결과 분석")

        # Scatter 1: 지하부 길이 vs 잎 수
        if "지하부 길이" in growth_df.columns and "잎 수" in growth_df.columns:
            plot_data = growth_df[["지하부 길이", "잎 수"]].dropna()
            if len(plot_data) >= 2:
                corr1 = plot_data["지하부 길이"].corr(plot_data["잎 수"])
                x1 = plot_data["지하부 길이"].values
                y1 = plot_data["잎 수"].values
                z1 = np.polyfit(x1, y1, 1)
                p1 = np.poly1d(z1)

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x1, y=y1, mode="markers", name="개체", marker=dict(color="#FFA500", size=8)))
                fig1.add_trace(go.Scatter(x=x1, y=p1(x1), mode="lines", name="추세선", line=dict(color="#D32F2F", width=2)))
                fig1.update_layout(
                    title="지하부 길이 vs 잎 수",
                    xaxis_title="지하부 길이 (mm)",
                    yaxis_title="잎 수 (장)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig1, width="stretch")

                if corr1 > 0.3:
                    st.write(f"상관계수: {corr1:.3f} → 양의 상관관계")
                elif corr1 > 0:
                    st.write(f"상관계수: {corr1:.3f} → 약한 양의 상관관계")
                else:
                    st.write(f"상관계수: {corr1:.3f} → 음의 상관관계")
            else:
                st.info("지하부 길이와 잎 수 데이터가 2개 미만입니다.")
        else:
            st.info("지하부 길이와 잎 수 열을 찾을 수 없습니다.")

        # Scatter 2: 지하부 길이 vs 지상부 길이
        if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
            plot_data2 = growth_df[["지하부 길이", "지상부 길이"]].dropna()
            if len(plot_data2) >= 2:
                corr2 = plot_data2["지하부 길이"].corr(plot_data2["지상부 길이"])
                x2 = plot_data2["지하부 길이"].values
                y2 = plot_data2["지상부 길이"].values
                z2 = np.polyfit(x2, y2, 1)
                p2 = np.poly1d(z2)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x2, y=y2, mode="markers", name="개체", marker=dict(color="#87CEEB", size=8)))
                fig2.add_trace(go.Scatter(x=x2, y=p2(x2), mode="lines", name="추세선", line=dict(color="#D32F2F", width=2)))
                fig2.update_layout(
                    title="지하부 길이 vs 지상부 길이",
                    xaxis_title="지하부 길이 (mm)",
                    yaxis_title="지상부 길이 (mm)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig2, width="stretch")

                if corr2 > 0.7:
                    st.write(f"상관계수: {corr2:.3f} → 매우 강한 양의 상관관계")
                elif corr2 > 0.3:
                    st.write(f"상관계수: {corr2:.3f} → 양의 상관관계")
                else:
                    st.write(f"상관계수: {corr2:.3f} → 약한 상관관계")
            else:
                st.info("지하부 길이와 지상부 길이 데이터가 2개 미만입니다.")
        else:
            st.info("지하부 길이와 지상부 길이 열을 찾을 수 없습니다.")

        # Bar: individual weight
        if "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
            weight_data = growth_df[["개체번호", "생중량"]].dropna()
            if len(weight_data) >= 1:
                sorted_w = weight_data["생중량"].sort_values(ascending=False)
                top3 = set(sorted_w.head(3).index)
                bottom3 = set(sorted_w.tail(3).index)

                fig3 = go.Figure()
                colors = ["#FFD700" if i in top3 else "#DC143C" if i in bottom3 else "#87CEEB" for i in weight_data.index]
                fig3.add_trace(go.Bar(x=weight_data["개체번호"], y=weight_data["생중량"], marker=dict(color=colors)))
                fig3.update_layout(
                    title="개체별 생중량 비교 (상위 3개: 노란색, 하위 3개: 빨간색)",
                    xaxis_title="개체",
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig3, width="stretch")
            else:
                st.info("생중량 데이터가 없습니다.")
        else:
            st.info("생중량 또는 개체번호 열을 찾을 수 없습니다.")

        # Data table
        if not growth_df.empty:
            st.dataframe(growth_df, width="stretch")

        # CSV download
        csv = growth_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 생육 데이터 CSV 다운로드",
            data=csv,
            file_name=f"{school_name}_생육데이터.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 3: 실생활 대입 ────────────────────────────────────────────────────
with tab3:
    try:
        st.subheader("실생활 대입: 스마트팜 생육 진단·관리")

        st.write("**핵심 인사이트**")
        if not growth_df.empty:
            if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
                temp = growth_df[["지하부 길이", "지상부 길이"]].dropna()
                if len(temp) >= 2:
                    c2 = temp["지하부 길이"].corr(temp["지상부 길이"])
                    if c2 > 0.7:
                        st.write(f"• 지상부 길이 ↔ 지하부 길이: 매우 강한 양의 상관관계 (r ≈ {c2:.2f}) → 지상부 길이를 보면 지하부 크기를 추정할 수 있음")
                    else:
                        st.write(f"• 지상부 길이 ↔ 지하부 길이: 양의 상관관계 (r ≈ {c2:.2f})")

            if "지하부 길이" in growth_df.columns and "잎 수" in growth_df.columns:
                temp = growth_df[["지하부 길이", "잎 수"]].dropna()
                if len(temp) >= 2:
                    c1 = temp["지하부 길이"].corr(temp["잎 수"])
                    if c1 > 0.3:
                        st.write(f"• 잎 수 ↔ 지하부 길이: 양의 상관관계 (r ≈ {c1:.2f})")
                    else:
                        st.write(f"• 잎 수 ↔ 지하부 길이: 약한 상관관계 (r ≈ {c1:.2f}) → 잎 수만으로 지하부 상태를 예측하기 어려움")

        st.write("**스마트팜 적용 방안**")
        st.write("• 지상부 길이 정기 측정으로 지하부 생육 상태 간접 진단")
        st.write("• 환경 데이터(온도·습도·EC·pH)와 생육 결과의 상관관계 모니터링")
        st.write("• 이상 환경 구간에서 생육 지연 패턴 파악 → 환경 자동 조절")
        st.write("• 잎 수·지상부 길이 기준 조기 경보 시스템 구축")

        st.info("**학생 프로젝트 확장 아이디어**\n\n• 여러 학교의 데이터 비교 분석 (학교별 환경 차이)\n• 기상 데이터와 생육 결과 간 상관관계 분석\n• 머신러닝을 이용한 지하부 생장 예측 모델 개발")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
