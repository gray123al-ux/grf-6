
import streamlit as st
import pandas as pd
import plotly.express as px
import math

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="Sort Movie Types",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Sort Movie Types")
st.caption("KOBIS 영화 데이터를 이용해 영화의 유형을 찾아보는 클러스터링 실험실")


DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

REQUIRED_COLUMNS = [
    "movieCd",
    "movieNm",
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

FEATURES = {
    "log_first_scrn": "개봉 스크린 수 (로그)",
    "log_total_audi": "총 관객 수 (로그)",
    "days_in_top10": "10위권에 머문 날수",
    "long_run_index": "롱런 지수"
}

ORIGINAL_FEATURES = [
    "first_scrn",
    "total_audi",
    "days_in_top10",
    "long_run_index"
]


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------

@st.cache_data
def load_data(url):

    df = pd.read_csv(url, encoding="utf-8-sig")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    total_movies = len(df)

    numeric_columns = [
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 로그 계산에 필요한 값은 양수여야 함
    df["log_first_scrn"] = df["first_scrn"].map(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )

    df["log_total_audi"] = df["total_audi"].map(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )

    # 롱런 지수 = 총 관객 / 첫 주 관객
    # 최대값은 20으로 제한
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # 필요한 값이 모두 있는 영화만 사용
    usable_df = df.dropna(
        subset=[
            "log_first_scrn",
            "log_total_audi",
            "days_in_top10",
            "long_run_index"
        ]
    ).copy()

    # 0 이하의 잘못된 값 제외
    usable_df = usable_df[
        (usable_df["first_scrn"] > 0)
        & (usable_df["total_audi"] > 0)
        & (usable_df["first_week_audi"] > 0)
    ].copy()

    return df, usable_df, total_movies


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

try:
    original_df, usable_df, total_movies = load_data(DATA_URL)

except Exception as e:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.write("데이터 주소와 CSV 열 이름을 확인해 주세요.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------

st.subheader("📊 데이터 요약")

col1, col2, col3 = st.columns(3)

col1.metric("전체 영화 수", f"{total_movies:,}편")
col2.metric("분석 가능한 영화", f"{len(usable_df):,}편")
col3.metric(
    "제외된 영화",
    f"{total_movies - len(usable_df):,}편"
)

st.caption(
    "필요한 분석 값이 없거나 스크린 수, 관객 수, 첫 주 관객 수가 "
    "0 이하인 영화는 분석에서 제외합니다."
)

if len(usable_df) < 3:
    st.error("클러스터링을 수행하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()


# --------------------------------------------------
# 분석할 속성 선택
# --------------------------------------------------

st.divider()
st.subheader("1. 분석할 속성 선택")

selected_features = st.multiselect(
    "영화 유형을 나눌 때 사용할 속성을 선택하세요. (최소 2개)",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    format_func=lambda x: FEATURES[x]
)

if len(selected_features) < 2:
    st.warning("클러스터링을 하려면 속성을 최소 2개 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# K-Means 클러스터링
# --------------------------------------------------

st.divider()
st.subheader("2. 영화 유형 분석")

n_clusters = 3

if len(usable_df) < n_clusters:
    st.error("그룹 수보다 분석 가능한 영화 수가 적습니다.")
    st.stop()

X = usable_df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

usable_df["cluster"] = model.fit_predict(X_scaled)

# 총 관객 평균이 높은 그룹부터 ㉮, ㉯, ㉰ 배정
cluster_order = (
    usable_df.groupby("cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

group_labels = ["㉮", "㉯", "㉰"]

cluster_to_group = {
    cluster: group_labels[i]
    for i, cluster in enumerate(cluster_order)
}

usable_df["group"] = usable_df["cluster"].map(
    cluster_to_group
)


# --------------------------------------------------
# 3. 2차원 산점도
# --------------------------------------------------

st.divider()
st.subheader("3. 2차원 산점도")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축(X축)",
        selected_features,
        index=0,
        format_func=lambda x: FEATURES[x]
    )

with axis_col2:
    default_y = 1 if len(selected_features) > 1 else 0

    y_feature = st.selectbox(
        "세로축(Y축)",
        selected_features,
        index=default_y,
        format_func=lambda x: FEATURES[x]
    )

fig_2d = px.scatter(
    usable_df,
    x=x_feature,
    y=y_feature,
    color="group",
    hover_name="movieNm",
    hover_data={
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "long_run_index": ":.2f",
        "group": True
    },
    category_orders={
        "group": group_labels
    },
    labels={
        x_feature: FEATURES[x_feature],
        y_feature: FEATURES[y_feature],
        "group": "영화 유형"
    },
    title="속성에 따른 영화 유형 분포"
)

fig_2d.update_traces(
    marker=dict(size=9, opacity=0.75)
)

fig_2d.update_layout(
    height=600,
    legend_title_text="영화 유형"
)

st.plotly_chart(
    fig_2d,
    use_container_width=True
)

st.info(
    "그래프에서 가까이 모여 있는 영화들은 선택한 속성의 특징이 "
    "서로 비슷한 영화들입니다."
)


# --------------------------------------------------
# 4. 3차원 산점도
# --------------------------------------------------

st.divider()
st.subheader("4. 3차원 산점도")

if len(selected_features) >= 3:

    three_features = selected_features[:3]

    fig_3d = px.scatter_3d(
        usable_df,
        x=three_features[0],
        y=three_features[1],
        z=three_features[2],
        color="group",
        hover_name="movieNm",
        category_orders={
            "group": group_labels
        },
        labels={
            three_features[0]: FEATURES[three_features[0]],
            three_features[1]: FEATURES[three_features[1]],
            three_features[2]: FEATURES[three_features[2]],
            "group": "영화 유형"
        },
        title="3차원 영화 유형 분포"
    )

    fig_3d.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig_3d.update_layout(
        height=700,
        scene=dict(
            aspectmode="cube"
        ),
        legend_title_text="영화 유형"
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )

    st.caption(
        "마우스로 그래프를 드래그하면 3차원 공간을 회전할 수 있습니다."
    )

else:
    st.info(
        "3차원 그래프를 보려면 분석 속성을 3개 이상 선택하세요."
    )


# --------------------------------------------------
# 5. 그룹별 특징 비교
# --------------------------------------------------

st.divider()
st.subheader("5. 그룹별 특징 비교")

summary = (
    usable_df.groupby("group")
    .agg(
        영화수=("movieNm", "count"),
        평균_개봉스크린수=("first_scrn", "mean"),
        평균_총관객수=("total_audi", "mean"),
        평균_10위권날수=("days_in_top10", "mean"),
        평균_롱런지수=("long_run_index", "mean")
    )
    .reindex(group_labels)
    .reset_index()
)

summary = summary.rename(
    columns={
        "group": "영화 유형",
        "영화수": "영화 수",
        "평균_개봉스크린수": "평균 개봉 스크린 수",
        "평균_총관객수": "평균 총 관객 수",
        "평균_10위권날수": "평균 10위권 날수",
        "평균_롱런지수": "평균 롱런 지수"
    }
)

st.dataframe(
    summary.style.format({
        "영화 수": "{:,.0f}",
        "평균 개봉 스크린 수": "{:,.0f}",
        "평균 총 관객 수": "{:,.0f}",
        "평균 10위권 날수": "{:.1f}",
        "평균 롱런 지수": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 6. 그룹별 대표 영화 TOP 5
# --------------------------------------------------

st.divider()
st.subheader("6. 그룹별 대표 영화 TOP 5")

for group in group_labels:

    group_df = usable_df[
        usable_df["group"] == group
    ].sort_values(
        "total_audi",
        ascending=False
    ).head(5)

    if group == "㉮":
        group_title = "㉮ 총 관객 수가 가장 높은 그룹"
    elif group == "㉯":
        group_title = "㉯ 두 번째 그룹"
    else:
        group_title = "㉰ 세 번째 그룹"

    with st.expander(group_title, expanded=True):

        if group_df.empty:
            st.write("해당 그룹에 영화가 없습니다.")
        else:
            display_df = group_df[
                [
                    "movieNm",
                    "genre",
                    "first_scrn",
                    "first_week_audi",
                    "total_audi",
                    "days_in_top10",
                    "long_run_index"
                ]
            ].rename(
                columns={
                    "movieNm": "영화명",
                    "genre": "장르",
                    "first_scrn": "개봉 스크린 수",
                    "first_week_audi": "첫 주 관객 수",
                    "total_audi": "총 관객 수",
                    "days_in_top10": "10위권 날수",
                    "long_run_index": "롱런 지수"
                }
            )

            st.dataframe(
                display_df.style.format({
                    "개봉 스크린 수": "{:,.0f}",
                    "첫 주 관객 수": "{:,.0f}",
                    "총 관객 수": "{:,.0f}",
                    "10위권 날수": "{:.0f}",
                    "롱런 지수": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.subheader("💡 분석 결과 생각해 보기")

st.markdown("""
- **㉮ 그룹**은 어떤 특징을 가진 영화들인가요?
- **㉯ 그룹**과 **㉰ 그룹**은 무엇이 다른가요?
- 개봉 스크린 수가 많으면 총 관객 수도 많을까요?
- 첫 주 관객이 적어도 롱런 지수가 높은 영화가 있을까요?
- 분석 속성을 바꾸면 영화 그룹은 어떻게 달라지나요?
""")

st.caption(
    "KOBIS 영화 데이터를 활용한 K-Means 클러스터링 실험"
)
