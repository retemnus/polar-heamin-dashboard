
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- 파일 읽기 ----------
def read_csv_bytes(data):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ---------- 열 이름 표준화 ----------
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
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    return df

# ---------- 데이터 로드 (캐시) ----------
@st.cache_data
def load_growth(raw_bytes, file_name):
    df = read_csv_bytes(raw_bytes)
    return std_cols(df), file_name

# ---------- 그래프 함수 ----------
BAR_TITLES = {
    "생중량": "개체별 평균 생중량 (g)",
    "잎 수": "개체별 평균 잎 수 (장)",
    "지상부 길이": "개체별 평균 지상부 길이 (mm)",
    "지하부 길이": "개체별 평균 지하부 길이 (mm)",
}

def plot_individual_bar(g, item):
    y = g[item].dropna()
    if y.empty:
        return None
    th_high = y.quantile(0.75)
    th_low = y.quantile(0.25)
    colors = []
    for v in y:
        if v >= th_high:
            colors.append("#E63946")
        elif v <= th_low:
            colors.append("#1D3557")
        else:
            colors.append("#8D99AE")
    fig = go.Figure(
        go.Bar(
            x=g.index.astype(str),
            y=y.values,
            marker_color=colors,
            text=y.round(2).astype(str),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=BAR_TITLES[item],
        xaxis_title="개체",
        yaxis_title=item,
        height=300,
        xaxis_tickangle=-45,
    )
    return fig

def make_bar_grid(g, items):
    n = len(items)
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[BAR_TITLES[i] for i in items])
    for idx, item in enumerate(items):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(plot_individual_bar(g, item).data[0], row=r, col=c)
    fig.update_layout(height=320 * rows, showlegend=False)
    return fig

def plot_scatter(g, x_col, y_col, x_label):
    d = g[[x_col, y_col]].dropna()
    if len(d) < 3:
        return None
    try:
        x_v = d[x_col].astype(float).values
        y_v = d[y_col].values
        corr = np.corrcoef(x_v, y_v)[0, 1]
        coef = np.polyfit(x_v, y_v, 1)
        line_x = np.linspace(x_v.min(), x_v.max(), 50)
        line_y = np.poly1d(coef)(line_x)
    except Exception:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_v,
            y=y_v,
            mode="markers",
            name="개체",
            marker=dict(color="#4C72B0", size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="추세선",
            line=dict(color="#C44E52", dash="dash"),
        )
    )
    step = max(1, int(np.ceil((x_v.max() - x_v.min()) / 8)))
    fig.update_layout(
        title=f"{x_col} vs {y_col} · 상관계수: {corr:.2f}",
        xaxis_title=x_label,
        yaxis_title="생중량(g)",
        height=420,
        xaxis=dict(
            tickvals=np.arange(np.floor(x_v.min()), np.ceil(x_v.max()) + 1, step),
            tickformat=",.0f",
        ),
    )
    return fig

# ---------- 앱 ----------
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

growth_file = st.sidebar.file_uploader("🌱 생육 결과 CSV", type="csv")
selected_items = st.sidebar.multiselect(
    "측정 항목 선택",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
)

if growth_file is not None:
    growth, school_name = load_growth(growth_file.getvalue(), growth_file.name)
    school_name = growth_file.name.rsplit(".", 1)[0].replace("_", " ").strip()
else:
    growth = None
    school_name = "우리 학교"

st.sidebar.info("측정 항목을 선택하면 해당 그래프만 표시해요.")

tab1, tab2 = st.tabs(["📖 실험 개요", "📊 생육 결과"])

# ---------- 탭 1: 실험 개요 ----------
with tab1:
    try:
        if growth is None or len(growth) == 0:
            st.info("🌱 생육 결과 CSV를 업로드하면 실험 개요를 볼 수 있어요.")
        else:
            st.subheader("연구 배경 및 목적")
            st.write(
                "극지식물은 추운 환경에서 짧은 기간에 빠르게 자라야 해요. "
                "같은 학교 안에서도 개체마다 잎 수, 줄기·뿌리의 길이, 무게가 서로 다르답니다. "
                "이 대시보드에서는 우리 학교 개체들의 생육 결과를 비교하고 주요 지표를 확인해요."
            )

            items = [
                c
                for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
                if c in growth.columns
            ]
            summary = pd.DataFrame(
                [
                    ["학교명", school_name],
                    ["개체 수", len(growth)],
                    ["측정 기간", "측정 기간 정보 없음"],
                    ["측정 항목", ", ".join(items) if items else "없음"],
                ],
                columns=["항목", "내용"],
            )
            st.write("**데이터 요약**")
            st.dataframe(summary, width="stretch")

            avg_w = growth["생중량"].dropna()
            if len(avg_w) < 2:
                st.info("측정된 생중량 값이 부족해요.")
            else:
                max_idx = avg_w.idxmax()
                min_idx = avg_w.idxmin()
                max_name = growth.loc[max_idx, "개체번호"]
                min_name = growth.loc[min_idx, "개체번호"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("총 개체수", f"{len(growth)} 개체")
                c2.metric("평균 생중량", f"{avg_w.mean():.2f} g")
                c3.metric("최대 생중량 개체", f"{max_name} ({growth.loc[max_idx, '생중량']:.2f} g)")
                c4.metric("최소 생중량 개체", f"{min_name} ({growth.loc[min_idx, '생중량']:.2f} g)")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------- 탭 2: 생육 결과 ----------
with tab2:
    try:
        if growth is None or len(growth) == 0:
            st.info("🌱 생육 결과 CSV를 업로드하면 생육 결과 분석을 볼 수 있어요.")
        else:
            w = growth["생중량"].dropna()
            if len(w) < 3:
                st.info("측정된 생중량 값이 부족해요.")
            else:
                st.subheader("핵심 결과")
                max_idx = w.idxmax()
                min_idx = w.idxmin()
                max_name = growth.loc[max_idx, "개체번호"]
                min_name = growth.loc[min_idx, "개체번호"]
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 생중량", f"{w.mean():.2f} g")
                c2.metric("최대 개체", f"{max_name} ({growth.loc[max_idx, '생중량']:.2f} g)")
                c3.metric("최소 개체", f"{min_name} ({growth.loc[min_idx, '생중량']:.2f} g)")

                # 상위 25% vs 하위 25%
                q75 = w.quantile(0.75)
                q25 = w.quantile(0.25)
                top = growth[growth["생중량"] >= q75]
                bottom = growth[growth["생중량"] <= q25]
                rows = []
                for label, subset in [("상위 25%", top), ("하위 25%", bottom)]:
                    row = {"구분": label, "개체 수": len(subset)}
                    for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]:
                        if c in subset.columns:
                            s = subset[c].dropna()
                            row[c] = round(s.mean(), 2) if len(s) else None
                        else:
                            row[c] = "—"
                    rows.append(row)
                st.write("**생중량 상위 25% vs 하위 25% 비교**")
                st.dataframe(pd.DataFrame(rows), width="stretch")

                # 생중량 요약 통계
                sd = w.std()
                cv = sd / w.mean() if w.mean() else 0
                st.write("**생중량 요약 통계**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "지표": ["평균", "표준편차", "변이계수(CV)"],
                            "값": [f"{w.mean():.2f} g", f"{sd:.2f} g", f"{cv:.2f}"],
                        }
                    ),
                    width="stretch",
                )

                # 개체별 생육 비교 (막대 그래프)
                num_cols = [c for c in selected_items if c in growth.columns]
                g = growth.groupby("개체번호").mean(numeric_only=True)
                g = g[[c for c in num_cols if c in g.columns]]
                bar_items = [i for i in selected_items if i in g.columns and i in BAR_TITLES]
                if bar_items:
                    st.write("**개체별 생육 비교**")
                    st.plotly_chart(make_bar_grid(g, bar_items), width="stretch")
                else:
                    st.info("선택한 측정 항목으로 개체별 비교를 할 수 없어요.")

                # 생중량 분포
                st.subheader("생중량 분포")
                fig = go.Figure(
                    go.Box(
                        y=w,
                        name="생중량",
                        boxpoints="outliers",
                        marker_color="#4C72B0",
                    )
                )
                fig.update_layout(
                    title="개체별 생중량 분포",
                    yaxis_title="생중량(g)",
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")

                # 상관관계 분석
                st.subheader("상관관계 분석")
                sc1 = plot_scatter(growth, "잎 수", "생중량", "잎 수(장)")
                sc2 = plot_scatter(growth, "지상부 길이", "생중량", "지상부 길이(mm)")
                if sc1 is not None:
                    st.plotly_chart(sc1, width="stretch")
                else:
                    st.info("잎 수 데이터가 부족해요.")
                if sc2 is not None:
                    st.plotly_chart(sc2, width="stretch")
                else:
                    st.info("지상부 길이 데이터가 부족해요.")

            with st.expander("📂 생육 데이터 원본 및 CSV 다운로드"):
                st.dataframe(growth, width="stretch")
                csv_bytes = growth.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
