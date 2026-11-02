
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Dividing Movie Types",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Dividing Movie Types")
st.caption(
    "Discover different movie types by grouping films "
    "with similar characteristics."
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
DATA_URL = (
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
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(url):

    df = pd.read_csv(url, dtype={"movieCd": str})

    total_movies = len(df)

    required_columns = [
        "movieNm",
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    numeric_columns = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    df = df[
        (df["first_scrn"] > 0) &
        (df["total_audi"] > 0) &
        (df["first_week_audi"] > 0) &
        (df["days_in_top10"] >= 0)
    ].copy()

    df = df.sort_values("movieCd").reset_index(drop=True)

    # Original attributes
    df["Number of Screens"] = df["first_scrn"]
    df["Cumulative Audience"] = df["total_audi"]
    df["Days in Top 10"] = df["days_in_top10"]

    df["Long Run Index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # FIX: Use NumPy instead of math.log10()
    df["log_first_scrn"] = np.log10(
        df["first_scrn"].where(df["first_scrn"] > 0)
    )

    df["log_total_audi"] = np.log10(
        df["total_audi"].where(df["total_audi"] > 0)
    )

    # Display-friendly attribute names
    df["Number of Screens (Log)"] = df["log_first_scrn"]
    df["Cumulative Audience (Log)"] = df["log_total_audi"]

    return df, total_movies


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
try:
    df, total_movies = load_data(DATA_URL)

except Exception as e:
    st.error(f"Unable to load movie data: {e}")
    st.stop()

if len(df) < 7:
    st.error("Not enough valid movie records for clustering.")
    st.stop()

st.info(
    f"Using **{len(df):,} movies** out of "
    f"**{total_movies:,} original records**."
)

# --------------------------------------------------
# CLUSTERING SETTINGS
# --------------------------------------------------
st.header("⚙️ Clustering Settings")

selected_attributes = st.multiselect(
    "Select attributes for grouping",
    options=ATTRIBUTE_LABELS,
    default=ATTRIBUTE_LABELS
)

if len(selected_attributes) < 2:
    st.warning("Please select at least two attributes.")
    st.stop()

k = st.slider(
    "Number of movie groups",
    min_value=2,
    max_value=7,
    value=3,
    step=1
)

# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------
X = df[selected_attributes].to_numpy()

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
    df.assign(cluster=cluster_numbers)
    .groupby("cluster")["Cumulative Audience"]
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
# OVERVIEW
# --------------------------------------------------
st.divider()
st.header("📊 Movie Group Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Movies", f"{len(df):,}")
c2.metric("Number of Groups", k)
c3.metric("Selected Attributes", f"{len(selected_attributes)} / 4")

# --------------------------------------------------
# 2D SCATTER PLOT
# --------------------------------------------------
st.divider()
st.header("🗺️ Bundle Map · 2D")

col1, col2 = st.columns(2)

x_axis = col1.selectbox(
    "Horizontal Axis",
    ATTRIBUTE_LABELS,
    index=0
)

y_axis = col2.selectbox(
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
    category_orders={
        "Bundle": active_symbols
    },
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_traces(
    marker=dict(size=9, opacity=0.8)
)

fig.update_layout(
    height=480,
    template="plotly_dark",
    legend_title="Movie Group"
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Each dot represents one movie. "
    "Change the axes to explore differences between groups."
)

# --------------------------------------------------
# 3D SCATTER PLOT
# --------------------------------------------------
st.divider()
st.header("🌐 Bundle Map · 3D")

if len(selected_attributes) < 3:

    st.info(
        "Select at least three attributes to display the 3D map."
    )

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

        st.warning(
            "Please select three different attributes."
        )

    else:

        fig3 = px.scatter_3d(
            df,
            x=x3,
            y=y3,
            z=z3,
            color="Bundle",
            hover_name="movieNm",
            category_orders={
                "Bundle": active_symbols
            },
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
            "Drag to rotate the 3D map. "
            "Hover over a point to see its movie title."
        )

# --------------------------------------------------
# GROUP STATISTICS
# --------------------------------------------------
st.divider()
st.header("📋 Group Statistics")

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
    "Statistics are calculated using original units. "
    "The Long Run Index is cumulative audience divided by "
    "first-week audience, capped at 20."
)

# --------------------------------------------------
# TOP 5 MOVIES IN EACH GROUP
# --------------------------------------------------
st.divider()
st.header("🏆 Top 5 Movies in Each Group")

columns = st.columns(k)

for i, symbol in enumerate(active_symbols):

    group = (
        df[df["Bundle"] == symbol]
        .sort_values(
            "Cumulative Audience",
            ascending=False
        )
        .head(5)
    )

    count = (df["Bundle"] == symbol).sum()

    with columns[i]:

        st.subheader(f"{symbol} Group")
        st.caption(f"{count:,} movies")

        display = group[
            ["movieNm", "Cumulative Audience"]
        ].rename(
            columns={
                "movieNm": "Movie",
                "Cumulative Audience": "Audience"
            }
        )

        st.dataframe(
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
# DOWNLOAD
# --------------------------------------------------
st.divider()
st.header("📥 Download Results")

csv = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    "Download Movie Group Data (CSV)",
    data=csv,
    file_name="movie_types.csv",
    mime="text/csv"
)

st.success("Movie clustering completed successfully!")
```
