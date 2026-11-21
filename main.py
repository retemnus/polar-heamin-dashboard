import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from plotly.subplots import make_subplots

st.set_page_config(page_title="극지식물 생육 상관관계 분석", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

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
def load_csv(file):
    if file is None:
        return pd.DataFrame()
    return std_cols(read_csv(file))

growth_df = pd.DataFrame()
if growth_file is not None:
    with st.spinner("생육 CSV를 읽는 중..."):
        growth_df = load_csv(growth_file)
else:
    st.info("사이드바에서 생육 CSV 파일을 업로드해 주세요.")

school_name = growth_file.name.split(".")[0] if growth_file is not None else ""

def analyze_relationship(x, y):
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        return None, None, None, None
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    r2 = r ** 2
    return slope, intercept, r, r2

def add_trendline(x, y, point_color="steelblue"):
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False))
        fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
        return fig, None, None, None, None
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    r2 = r ** 2
    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_clean, y=y_clean,
        mode="markers",
        name="개체 데이터",
        marker=dict(color=point_color, size=8, opacity=0.7)
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=slope * x_range + intercept,
        mode="lines",
        name=f"추세선 (r={r:.2f}, R²={r2:.2f})",
        line=dict(color="red", dash="dash", width=2)
    ))
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig, slope, intercept, r, r2

def interpret_r(r):
    if r < 0.3:
        return "매우 약한 양의 상관관계"
    elif r < 0.7:
        return "중간 정도의 양의 상관관계"
    else:
        return "강한 양의 상관관계"

def format_eq(slope, intercept, unit=""):
    return f"지하부 길이 = {slope:.3f} × {unit} + {intercept:.3f}"

if growth_df.empty:
    st.info("생육 결과 CSV 파일을 업로드하면 분석 결과를 볼 수 있어요.")
else:
    tab1, tab2, tab3 = st.tabs(["실험 소개", "생육 결과", "실생활 대입"])

    with tab1:
        try:
            st.header("🌱 지상부 요소로 지하부 상태 예측하기")
            st.write("극지식물의 지상부 특성을 관찰하여 지하부 생육 상태를 예측할 수 있을까요?")

            st.subheader("연구의 핵심 질문")
            st.info("지상부의 어떤 요소를 보고 지하부의 상태를 예측할 수 있을까?")

            st.subheader("연구 가설 및 목적")
            c1, c2 = st.columns(2)
            with c1:
                st.info("**가설**: 지상부 생육이 좋은 개체일수록 지하부 생육도 활발할 것이다.")
            with c2:
                st.info("**목적**: 지상부 지표만으로 지하부 상태를 예측하고 비파괴적 생육 진단 가능성을 탐색해요.")

            avg_w = growth_df["생중량"].dropna().mean() if "생중량" in growth_df.columns else growth_df["지상부 질량"].dropna().mean() + growth_df["지하부 질량"].dropna().mean()
            avg_shoot = growth_df["지상부 길이"].dropna().mean()
            avg_root = growth_df["지하부 길이"].dropna().mean()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("전체 개체 수", f"{len(growth_df)}개체")
            with c2:
                st.metric("평균 생중량", f"{avg_w:.2f} g")
            with c3:
                st.metric("평균 지상부 길이", f"{avg_shoot:.2f} mm")
            with c4:
                st.metric("평균 지하부 길이", f"{avg_root:.2f} mm")

            st.subheader("주요 분석 결과")
            if "잎 수" in growth_df.columns and "지하부 길이" in growth_df.columns:
                a1, b1, r1, r2_1 = analyze_relationship(growth_df["잎 수"], growth_df["지하부 길이"])
                if a1 is not None:
                    st.write(f"**잎 수 vs 지하부 길이**: 상관계수 r = `{r1:.3f}`, 결정계수 R² = `{r2_1:.3f}`")
                    st.write(f"**회귀식**: 지하부 길이 = {a1:.3f} × 잎 수 + {b1:.3f}")
                    st.write(f"**해석**: 잎 수와 지하부 길이의 관계는 {interpret_r(r1)} (r = {r1:.3f}).")

            if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                a2, b2, r2, r2_2 = analyze_relationship(growth_df["지상부 길이"], growth_df["지하부 길이"])
                if a2 is not None:
                    st.write(f"**지상부 길이 vs 지하부 길이**: 상관계수 r = `{r2:.3f}`, 결정계수 R² = `{r2_2:.3f}`")
                    st.write(f"**회귀식**: 지하부 길이 = {a2:.3f} × 지상부 길이 + {b2:.3f}")
                    st.write(f"**해석**: 지상부 길이와 지하부 길이의 관계는 {interpret_r(r2)} (r = {r2:.3f}).")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tab2:
        try:
            st.header("📊 생육 결과 분석")
            st.write(f"총 {len(growth_df)}개체, 열: {', '.join(growth_df.columns.tolist())}")

            if "생중량" not in growth_df.columns:
                st.info("생중량 열이 없어 지상부 질량과 지하부 질량을 더해 계산했어요.")

            st.subheader("지하부 길이 vs 잎 수")
            fig1, a1, b1, r1, r2_1 = add_trendline(growth_df["잎 수"], growth_df["지하부 길이"], point_color="seagreen")
            st.plotly_chart(fig1, width="stretch")
            if a1 is not None:
                st.write(f"**회귀식**: 지하부 길이 = {a1:.3f} × 잎 수 + {b1:.3f}")
                st.write(f"**상관계수**: r = {r1:.3f}, 결정계수 R² = {r2_1:.3f}")
                st.write(f"**해석**: 잎 수와 지하부 길이의 관계는 {interpret_r(r1)} (r = {r1:.3f}).")

            st.divider()

            st.subheader("지하부 길이 vs 지상부 길이")
            fig2, a2, b2, r2, r2_2 = add_trendline(growth_df["지상부 길이"], growth_df["지하부 길이"], point_color="orange")
            st.plotly_chart(fig2, width="stretch")
            if a2 is not None:
                st.write(f"**회귀식**: 지하부 길이 = {a2:.3f} × 지상부 길이 + {b2:.3f}")
                st.write(f"**상관계수**: r = {r2:.3f}, 결정계수 R² = {r2_2:.3f}")
                st.write(f"**해석**: 지상부 길이와 지하부 길이의 관계는 {interpret_r(r2)} (r = {r2:.3f}).")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    with tab3:
        try:
            st.header("🏠 실생활 대입")
            st.subheader("핵심 인사이트")
            if "잎 수" in growth_df.columns and "지하부 길이" in growth_df.columns:
                a1, b1, r1, r2_1 = analyze_relationship(growth_df["잎 수"], growth_df["지하부 길이"])
            if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                a2, b2, r2, r2_2 = analyze_relationship(growth_df["지상부 길이"], growth_df["지하부 길이"])

            st.write("- **지상부·지하부 상관관계**: 지상부 길이가 길수록 지하부 길이도 길어져요. 식물의 지상부와 지하부는 함께 자라는 경향이 있어요.")
            if r2 is not None and r2 >= 0.7:
                st.write(f"- **강한 양의 상관관계 확인**: 지상부 길이와 지하부 길이의 상관계수는 **{r2:.2f}** (R² = {r2_2:.2f})로 매우 강한 양의 상관관계를 보여요.")
            elif r2 is not None:
                st.write(f"- **상관관계 확인**: 지상부 길이와 지하부 길이의 상관계수는 **{r2:.2f}** (R² = {r2_2:.2f})로 양의 상관관계가 있어요.")

            st.subheader("스마트팜 작물 생육 진단·관리 대입")
            st.write("""
            1. **지상부 지표 기반 지하부 추정**: 잎 수, 지상부 길이 등 비파괴적으로 측정 가능한 지표로 지하부 상태를 추정하여 정밀 관리해요.
            2. **생체중 예측 활용**: 지상부 질량·길이 데이터와 지하부 질량 간 상관관계를 활용해 수확 전 생체중을 예측해요.
            3. **생육 이상 조기 발견**: 지상부 성장만 과도하게 진행되고 지하부 성장이 저조하면 뿌리 병해·영양분 부족 등을 조기에 진단해요.
            4. **수확 시기 판단**: 지하부 길이가 일정 성숙 단계에 도달하면 지상부 상태와 함께 수확 적정 시점을 판단해요.
            """)

            st.subheader("📈 수확에서의 실생활 대입")
            st.write("""
            - 수확 전 지상부(잎·줄기) 길이와 무게를 측정하여 지하부(뿌리) 무게까지 추정해요.
            - 수출·출하 전에 작물 전체의 시장성(생체중 기준)을 예측해요.
            - 환경 데이터(온도·습도·EC 등)와 생육 지표를 연계하여 최적 재배 조건을 도출해요.
            """)
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    if not growth_df.empty:
        csv = growth_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 생육 데이터 CSV 다운로드 (한글 깨짐 방지)",
            data=csv,
            file_name=f"{school_name or '극지식물'}_생육분석.csv",
            mime="text/csv"
        )
