import math
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Sort Movie Types",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

RANDOM_STATE = 42

st.markdown("""
<style>
.main-title {
    font-size: 2.7rem;
    font-weight: 800;
    padding-bottom: 0;
}
.subtitle {
    color: #9CA3AF;
    font-size: 1rem;
    margin-bottom: 25px;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. TITLE
# ==========================================
st.markdown(
    '<div class="main-title">🎬 Sort Movie Types</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Group similar movies by opening scale and box-office longevity.'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "영화의 개봉 규모와 흥행 지속성을 바탕으로 "
    "비슷한 영화끼리 묶어 봅니다."
)


# ==========================================
# 3. LOAD DATA
# ==========================================
@st.cache_data
def load_data(url):

    data = pd.read_csv(url, encoding="utf-8")

    required = [
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

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {missing}"
        )

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

    total_movies = len(data)

    # Make a working copy
    df = data.copy()

    # Common logarithm of opening screen count
    df["log_first_scrn"] = df["first_scrn"].apply(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )

    # Common logarithm of cumulative audience
    df["log_total_audi"] = df["total_audi"].apply(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )

    # Long-run index
    df["long_run_index"] = (
        df["total_audi"] /
        df["first_week_audi"].where(
            df["first_week_audi"] > 0
        )
    ).clip(upper=20)

    attributes = [
        "log_first_scrn",
        "log_total_audi",
        "days_in_top10",
        "long_run_index"
    ]

    # Exclude movies with missing or invalid attributes
    df = df.dropna(subset=attributes)

    df = df[
        (df["first_week_audi"] > 0) &
        (df["first_scrn"] > 0) &
        (df["total_audi"] > 0)
    ].copy()

    df["movieNm"] = (
        df["movieNm"]
        .fillna("Unknown Movie")
        .astype(str)
    )

    return data, df, total_movies


try:
    original_df, df, total_movies = load_data(DATA_URL)

except Exception as error:
    st.error(
        "데이터를 불러오거나 처리하지 못했습니다. "
        "데이터 주소와 CSV 열 이름을 확인해 주세요."
    )

    st.exception(error)
    st.stop()


# ==========================================
# 4. ATTRIBUTE INFORMATION
# ==========================================
attributes = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "long_run_index"
]

attribute_names = {
    "log_first_scrn": "Log10(Number of Screens)",
    "log_total_audi": "Log10(Cumulative Audience)",
    "days_in_top10": "Days in Top 10",
    "long_run_index": "Long-run Index"
}

valid_count = len(df)
excluded_count = total_movies - valid_count


# ==========================================
# 5. DATA SUMMARY
# ==========================================
st.subheader("📊 Dataset Overview")

st.markdown(
    f"**Total movies: {total_movies:,}**　|　"
    f"**Movies available for grouping: {valid_count:,}**"
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", f"{total_movies:,}")
col2.metric("Valid Movies", f"{valid_count:,}")
col3.metric("Excluded Movies", f"{excluded_count:,}")

st.caption(
    "Movies with missing or invalid attributes, "
    "or zero first-week audience, are excluded."
)

st.divider()

if valid_count < 3:
    st.error(
        "At least three valid movies are required "
        "for three-group clustering."
    )
    st.stop()


# ==========================================
# 6. SELECT ATTRIBUTES
# ==========================================
st.subheader("1. Choose Clustering Attributes")

selected = st.multiselect(
    "Select at least two attributes:",
    options=attributes,
    default=attributes,
    format_func=lambda x: attribute_names[x]
)

if len(selected) < 2:
    st.warning("Please select at least two attributes.")
    st.stop()


# ==========================================
# 7. K-MEANS CLUSTERING
# ==========================================
X = df[selected].astype(float)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    kmeans = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================
# 8. RANK GROUPS
# ==========================================
cluster_order = (
    df.groupby("cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

symbols = ["㉮", "㉯", "㉰"]

group_mapping = {
    cluster: symbols[index]
    for index, cluster in enumerate(cluster_order)
}

df["Group"] = df["cluster"].map(group_mapping)

actual_groups = df["cluster"].nunique()

st.success(
    f"Clustering complete: {valid_count:,} movies "
    f"were processed into {actual_groups} groups."
)

if actual_groups < 3:
    st.warning(
        "Fewer than three distinct groups were produced. "
        "Some movies have very similar attribute values."
    )


# ==========================================
# 9. 2D SCATTER PLOT
# ==========================================
st.divider()

st.subheader("2. Two-dimensional Movie Map")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "Horizontal axis (X)",
        selected,
        index=0,
        format_func=lambda x: attribute_names[x],
        key="axis_x"
    )

with col2:
    y_axis = st.selectbox(
        "Vertical axis (Y)",
        selected,
        index=1,
        format_func=lambda x: attribute_names[x],
        key="axis_y"
    )

fig2d = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="Group",
    category_orders={
        "Group": symbols
    },
    hover_name="movieNm",
    hover_data=[
        "Group",
        "first_scrn",
        "total_audi",
        "first_week_audi",
        "days_in_top10",
        "long_run_index"
    ],
    labels={
        x_axis: attribute_names[x_axis],
        y_axis: attribute_names[y_axis],
        "Group": "Movie Group",
        "movieNm": "Movie Title"
    },
    title="Movie Type Distribution"
)

fig2d.update_traces(
    marker=dict(
        size=9,
        opacity=0.8
    )
)

fig2d.update_layout(
    height=570,
    legend_title="Movie Groups",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig2d,
    use_container_width=True
)


# ==========================================
# 10. 3D SCATTER PLOT
# ==========================================
st.divider()

st.subheader("3. Three-dimensional Movie Map")

if len(selected) < 3:

    st.info(
        "💡 Select at least three attributes above "
        "to display the 3D movie map."
    )

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        x3 = st.selectbox(
            "3D X axis",
            selected,
            index=0,
            format_func=lambda x: attribute_names[x],
            key="axis_3d_x"
        )

    with col2:
        y3 = st.selectbox(
            "3D Y axis",
            selected,
            index=1,
            format_func=lambda x: attribute_names[x],
            key="axis_3d_y"
        )

    with col3:
        z3 = st.selectbox(
            "3D Z axis",
            selected,
            index=2,
            format_func=lambda x: attribute_names[x],
            key="axis_3d_z"
        )

    fig3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="Group",
        category_orders={
            "Group": symbols
        },
        hover_name="movieNm",
        hover_data=[
            "Group",
            "first_scrn",
            "total_audi",
            "first_week_audi",
            "days_in_top10",
            "long_run_index"
        ],
        labels={
            x3: attribute_names[x3],
            y3: attribute_names[y3],
            z3: attribute_names[z3],
            "Group": "Movie Group"
        },
        title="3D Movie Type Distribution"
    )

    fig3d.update_traces(
        marker=dict(
            size=3,
            opacity=0.8
        )
    )

    fig3d.update_layout(
        height=700,
        scene=dict(
            aspectmode="cube",
            xaxis_title=attribute_names[x3],
            yaxis_title=attribute_names[y3],
            zaxis_title=attribute_names[z3]
        ),
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        )
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True,
        config={
            "scrollZoom": True
        }
    )


# ==========================================
# 11. GROUP STATISTICS
# ==========================================
st.divider()

st.subheader("4. Compare Movie Groups")

summary = (
    df.groupby("Group")
    .agg(
        Movie_Count=("movieNm", "count"),
        Average_Screens=("first_scrn", "mean"),
        Average_Total_Audience=("total_audi", "mean"),
        Average_Days_Top10=("days_in_top10", "mean"),
        Average_Long_Run_Index=("long_run_index", "mean")
    )
    .reindex(symbols)
    .dropna(how="all")
    .reset_index()
)

summary = summary.rename(
    columns={
        "Group": "Group",
        "Movie_Count": "Number of Movies",
        "Average_Screens": "Average Screens",
        "Average_Total_Audience": "Average Cumulative Audience",
        "Average_Days_Top10": "Average Days in Top 10",
        "Average_Long_Run_Index": "Average Long-run Index"
    }
)

st.caption(
    "The following averages use original units, "
    "not logarithms or standardized values."
)

st.dataframe(
    summary.style.format({
        "Number of Movies": "{:,.0f}",
        "Average Screens": "{:,.1f}",
        "Average Cumulative Audience": "{:,.0f}",
        "Average Days in Top 10": "{:,.1f}",
        "Average Long-run Index": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 12. TOP FIVE MOVIES IN EACH GROUP
# ==========================================
st.divider()

st.subheader("5. Top Five Movies in Each Group")

st.caption(
    "Movies are ranked by cumulative audience, "
    "from highest to lowest."
)

group_columns = st.columns(3)

for index, symbol in enumerate(symbols):

    group_movies = (
        df[df["Group"] == symbol]
        .sort_values(
            "total_audi",
            ascending=False
        )
        .head(5)
    )

    with group_columns[index]:

        st.markdown(f"### {symbol}")

        if group_movies.empty:

            st.caption("No movies in this group.")

        else:

            for rank, (_, movie) in enumerate(
                group_movies.iterrows(),
                start=1
            ):

                st.markdown(
                    f"**{rank}. {movie['movieNm']}**"
                )

                st.caption(
                    f"Cumulative audience: "
                    f"{movie['total_audi']:,.0f}"
                )


# ==========================================
# 13. FOOTER
# ==========================================
st.divider()

st.caption("""
### Clustering Method

- Algorithm: K-means
- Number of groups: 3
- Standardization: StandardScaler
- Random state: 42
- Initializations: 10
- Screen count: log10(first_scrn)
- Cumulative audience: log10(total_audi)
- Days in Top 10: Original value
- Long-run index: total_audi / first_week_audi
- Maximum long-run index: 20
- Group labels: ㉮, ㉯, ㉰
- Groups ranked by average cumulative audience
""")
