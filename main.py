
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Dividing Movie Types",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Dividing Movie Types")
st.caption(
    "Discover different movie types by grouping films "
    "with similar characteristics, without a fixed answer column."
)

# --------------------------------------------------
# Settings
# --------------------------------------------------
MOVIES_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

ATTRIBUTE_LABELS = [
    "Number of Screens (Log)",
    "Cumulative Audience (Log)",
    "Days in Top 10",
    "Long Run Index"
]

ORIGINAL_COLUMNS = [
    "Number of Screens",
    "Cumulative Audience",
    "Days in Top 10",
    "Long Run Index"
]

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES_URL, dtype={"movieCd": str})

    total_movies = len(df)

    required_columns = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi",
        "movieNm"
    ]

    df = df.dropna(subset=required_columns).copy()

    numeric_columns = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns)

    df = df[
        (df["first_scrn"] > 0) &
        (df["total_audi"] > 0) &
        (df["first_week_audi"] > 0) &
        (df["days_in_top10"] >= 0)
    ].copy()

    df = df.sort_values("movieCd").reset_index(drop=True)

    return df, total_movies


try:
    df, total_movies = load_data()
except Exception as e:
    st.error(f"Could not load the movie data: {e}")
    st.stop()

if len(df) < 7:
    st.error("Not enough movies to perform clustering.")
    st.stop()

# --------------------------------------------------
# Create four attributes
# --------------------------------------------------
df["Number of Screens"] = df["first_scrn"]
df["Cumulative Audience"] = df["total_audi"]
df["Days in Top 10"] = df["days_in_top10"]

df["Long Run Index"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

df["Number of Screens (Log)"] = np.log10(df["first_scrn"])
df["Cumulative Audience (Log)"] = np.log10(df["total_audi"])

st.info(
    f"Analyzing **{len(df):,} movies** out of "
    f"**{total_movies:,} original records**. "
    "Movies without the required data are excluded."
)

# --------------------------------------------------
# Attribute selection
# --------------------------------------------------
st.subheader("⚙️ Clustering Settings")

selected_attributes = st.multiselect(
    "Select attributes for grouping (at least two)",
    ATTRIBUTE_LABELS,
    default=ATTRIBUTE_LABELS
)

if len(selected_attributes) < 2:
    st.warning("Please select at least two attributes.")
    st.stop()

# --------------------------------------------------
# Number of clusters
# --------------------------------------------------
k = st.slider(
    "Number of movie groups",
    min_value=2,
    max_value=7,
    value=3,
    help="Choose how many different movie types to discover."
)

# --------------------------------------------------
# K-means clustering
# --------------------------------------------------
X = df[selected_attributes].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
)

cluster_numbers = model.fit_predict(X_scaled)

# Sort groups by average cumulative audience
cluster_order = (
    df.assign(_cluster=cluster_numbers)
    .groupby("_cluster")["Cumulative Audience"]
    .mean()
    .sort_values(ascending=False)
    .index
)

cluster_symbols = {
    cluster: SYMBOLS[i]
    for i, cluster in enumerate(cluster_order)
}

df["Bundle"] = [
    cluster_symbols[number]
    for number in cluster_numbers
]

active_symbols = SYMBOLS[:k]

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
st.divider()
st.subheader("📊 Movie Group Overview")

metric_columns = st.columns(3)

metric_columns[0].metric("Total Movies", f"{len(df):,}")
metric_columns[1].metric("Number of Groups", k)
metric_columns[2].metric(
    "Selected Attributes",
    f"{len(selected_attributes)} / 4"
)

# --------------------------------------------------
# 2D scatter plot
# --------------------------------------------------
st.divider()
st.subheader("🗺️ Bundle Map · 2D")

c1, c2 = st.columns(2)

x_axis = c1.selectbox(
    "Horizontal Axis",
    ATTRIBUTE_LABELS,
    index=0
)

y_axis = c2.selectbox(
    "Vertical Axis",
    ATTRIBUTE_LABELS,
    index=1
)

fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="Bundle",
    hover_name="movieNm",
    hover_data={
        "Number of Screens": ":,",
        "Cumulative Audience": ":,",
        "Days in Top 10": True,
        "Long Run Index": ":.2f"
    },
    category_orders={"Bundle": active_symbols},
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_traces(
    marker=dict(size=9, opacity=0.8)
)

fig.update_layout(
    height=480,
    legend_title="Movie Group",
    template="plotly_dark"
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Each dot represents one movie. "
    "Change the axes to explore how the groups differ."
)

# --------------------------------------------------
# 3D scatter plot
# --------------------------------------------------
st.divider()
st.subheader("🌐 Bundle Map · 3D")

if len(selected_attributes) < 3:
    st.info("Select at least three attributes to display the 3D plot.")
else:
    d1, d2, d3 = st.columns(3)

    x3 = d1.selectbox(
        "X Axis",
        selected_attributes,
        index=0,
        key="x3"
    )

    y3 = d2.selectbox(
        "Y Axis",
        selected_attributes,
        index=1,
        key="y3"
    )

    z3 = d3.selectbox(
        "Z Axis",
        selected_attributes,
        index=2,
        key="z3"
    )

    if len({x3, y3, z3}) < 3:
        st.warning("Please select three different attributes.")
    else:
        fig3 = px.scatter_3d(
            df,
            x=x3,
            y=y3,
            z=z3,
            color="Bundle",
            hover_name="movieNm",
            category_orders={"Bundle": active_symbols},
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig3.update_traces(
            marker=dict(size=3, opacity=0.75)
        )

        fig3.update_layout(
            height=600,
            template="plotly_dark",
            legend_title="Movie Group"
        )

        st.plotly_chart(fig3, width="stretch")

        st.caption(
            "Drag to rotate the 3D chart. "
            "Hover over a point to see the movie title."
        )

# --------------------------------------------------
# Group statistics
# --------------------------------------------------
st.divider()
st.subheader("📋 Group Statistics")

summary = (
    df.groupby("Bundle")[ORIGINAL_COLUMNS]
    .mean()
    .round(2)
)

summary.insert(
    0,
    "Number of Movies",
    df.groupby("Bundle").size()
)

summary = summary.reindex(active_symbols)

st.dataframe(
    summary,
    width="stretch",
    column_config={
        "Number of Movies": st.column_config.NumberColumn(
            "Number of Movies",
            format="%,d"
        ),
        "Number of Screens": st.column_config.NumberColumn(
            "Average Screens",
            format="%,.1f"
        ),
        "Cumulative Audience": st.column_config.NumberColumn(
            "Average Cumulative Audience",
            format="%,.0f"
        ),
        "Days in Top 10": st.column_config.NumberColumn(
            "Average Days in Top 10",
            format="%.1f"
        ),
        "Long Run Index": st.column_config.NumberColumn(
            "Average Long Run Index",
            format="%.2f"
        )
    }
)

st.caption(
    "Values are displayed in their original units, "
    "before logarithmic transformation. "
    "The Long Run Index is cumulative audience divided by "
    "first-week audience, capped at 20."
)

# --------------------------------------------------
# Top 5 movies in each group
# --------------------------------------------------
st.divider()
st.subheader("🏆 Top 5 Movies in Each Group")

columns = st.columns(k)

for i, symbol in enumerate(active_symbols):

    group = (
        df[df["Bundle"] == symbol]
        .sort_values("Cumulative Audience", ascending=False)
        .head(5)
    )

    count = (df["Bundle"] == symbol).sum()

    columns[i].markdown(
        f"### {symbol} Group"
    )

    columns[i].caption(
        f"{count:,} movies"
    )

    display = group[
        ["movieNm", "Cumulative Audience"]
    ].rename(
        columns={
            "movieNm": "Movie",
            "Cumulative Audience": "Audience"
        }
    )

    columns[i].dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Audience": st.column_config.NumberColumn(
                "Audience",
                format="%,d"
            )
        }
    )

# --------------------------------------------------
# Download results
# --------------------------------------------------
st.divider()
st.subheader("📥 Download Results")

csv = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="Download Grouped Movie Data (CSV)",
    data=csv,
    file_name="movie_types.csv",
    mime="text/csv"
)

st.success(
    "Movie clustering completed successfully!"
)
