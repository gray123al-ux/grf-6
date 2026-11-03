
import math
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Sort Movie Types",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

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
    .group-card {
        padding: 18px;
        border-radius: 12px;
        background: rgba(128,128,128,0.08);
        margin-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================
st.markdown(
    '<div class="main-title">🎬 Sort Movie Types</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Discover movie patterns using K-means clustering.</div>',
    unsafe_allow_html=True
)


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd", "movieNm", "first_scrn",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    return data


try:
    original_df = load_data()
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()


# ==========================================
# CREATE FOUR ATTRIBUTES
# ==========================================
df = original_df.copy()

df["log_first_scrn"] = df["first_scrn"].apply(
    lambda x: math.log10(x) if pd.notna(x) and x > 0 else None
)

df["log_total_audi"] = df["total_audi"].apply(
    lambda x: math.log10(x) if pd.notna(x) and x > 0 else None
)

df["long_run_index"] = (
    df["total_audi"] / df["first_week_audi"].where(
        df["first_week_audi"] > 0
    )
).clip(upper=20)

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

# Exclude missing or invalid data
df = df.dropna(subset=attributes)
df = df[
    (df["first_week_audi"] > 0) &
    (df["first_scrn"] > 0) &
    (df["total_audi"] > 0)
].copy()

df["movieNm"] = df["movieNm"].fillna("Unknown Movie").astype(str)

total_count = len(original_df)
valid_count = len(df)


# ==========================================
# DATA SUMMARY
# ==========================================
st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Movies", f"{total_count:,}")
c2.metric("Valid Movies", f"{valid_count:,}")
c3.metric("Excluded Movies", f"{total_count-valid_count:,}")

st.divider()

if valid_count < 3:
    st.error("At least three valid movies are required.")
    st.stop()


# ==========================================
# SELECT ATTRIBUTES
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
# K-MEANS CLUSTERING
# ==========================================
X = df[selected].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)


# Rank groups by average cumulative audience
cluster_order = (
    df.groupby("cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

symbols = ["㉮", "㉯", "㉰"]

group_mapping = {
    cluster: symbols[i]
    for i, cluster in enumerate(cluster_order)
}

df["Group"] = df["cluster"].map(group_mapping)

st.success(
    f"Successfully grouped {len(df):,} movies into 3 groups."
)

if df["cluster"].nunique() < 3:
    st.warning(
        "Some clusters contain identical or insufficiently distinct "
        "patterns. Check the scatter plots and group table."
    )


# ==========================================
# 2D SCATTER PLOT
# ==========================================
st.divider()
st.subheader("2. Two-dimensional Movie Map")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "Horizontal axis (X)",
        selected,
        index=0,
        format_func=lambda x: attribute_names[x]
    )

with col2:
    y_axis = st.selectbox(
        "Vertical axis (Y)",
        selected,
        index=1,
        format_func=lambda x: attribute_names[x]
    )

fig2d = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="Group",
    category_orders={"Group": symbols},
    hover_name="movieNm",
    hover_data={
        "Group": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "long_run_index": ":.2f"
    },
    labels={
        x_axis: attribute_names[x_axis],
        y_axis: attribute_names[y_axis],
        "Group": "Movie Group"
    },
    title="Movie Type Distribution"
)

fig2d.update_traces(
    marker=dict(size=9, opacity=0.8)
)

fig2d.update_layout(
    height=570,
    legend_title="Movie Groups"
)

st.plotly_chart(fig2d, use_container_width=True)


# ==========================================
# 3D SCATTER PLOT
# ==========================================
st.divider()
st.subheader("3. Three-dimensional Movie Map")

if len(selected) < 3:

    st.info(
        "💡 Select at least three attributes above "
        "to display the 3D movie map."
    )

else:

    a, b, c = st.columns(3)

    with a:
        x3 = st.selectbox(
            "3D X axis",
            selected,
            index=0,
            format_func=lambda x: attribute_names[x]
        )

    with b:
        y3 = st.selectbox(
            "3D Y axis",
            selected,
            index=1,
            format_func=lambda x: attribute_names[x]
        )

    with c:
        z3 = st.selectbox(
            "3D Z axis",
            selected,
            index=2,
            format_func=lambda x: attribute_names[x]
        )

    fig3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="Group",
        category_orders={"Group": symbols},
        hover_name="movieNm",
        hover_data={
            "Group": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "long_run_index": ":.2f"
        },
        labels={
            x3: attribute_names[x3],
            y3: attribute_names[y3],
            z3: attribute_names[z3],
            "Group": "Movie Group"
        },
        title="3D Movie Type Distribution"
    )

    fig3d.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3d.update_layout(
        height=700,
        scene=dict(
            aspectmode="cube",
            xaxis_title=attribute_names[x3],
            yaxis_title=attribute_names[y3],
            zaxis_title=attribute_names[z3]
        )
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True,
        config={"scrollZoom": True}
    )


# ==========================================
# GROUP STATISTICS
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

summary = summary.rename(columns={
    "Group": "Group",
    "Movie_Count": "Number of Movies",
    "Average_Screens": "Average Screens",
    "Average_Total_Audience": "Average Cumulative Audience",
    "Average_Days_Top10": "Average Days in Top 10",
    "Average_Long_Run_Index": "Average Long-run Index"
})

st.dataframe(
    summary.style.format({
        "Number of Movies": "{:,.0f}",
        "Average Screens": "{:,.1f}",
        "Average Cumulative Audience": "{:,.0f}",
        "Average Days in Top 10": "{:.1f}",
        "Average Long-run Index": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# TOP FIVE MOVIES
# ==========================================
st.divider()
st.subheader("5. Top Five Movies in Each Group")

group_cols = st.columns(3)

for i, symbol in enumerate(symbols):

    group_movies = (
        df[df["Group"] == symbol]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with group_cols[i]:

        st.markdown(f"### {symbol}")

        if group_movies.empty:
            st.caption("No movies in this group.")

        else:
            for rank, (_, movie) in enumerate(
                group_movies.iterrows(), start=1
            ):

                st.markdown(
                    f"**{rank}. {movie['movieNm']}**"
                )

                st.caption(
                    f"Audience: {movie['total_audi']:,.0f}"
                )


# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption("""
**Clustering Method**

- Number of groups: 3
- Algorithm: K-means
- Standardization: StandardScaler
- Random state: 42
- Number of initializations: 10
- Screen count and cumulative audience: log10
- Long-run index: cumulative audience / first-week audience, maximum 20
- Group symbols ranked by average cumulative audience
""")
