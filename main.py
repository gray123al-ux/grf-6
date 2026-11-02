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
# LOAD AND PREPARE DATA
# Returns:
#   original_df: all rows from the CSV
#   usable_df: rows with all required valid attributes
#   total_movies: number of rows in the original CSV
# --------------------------------------------------
@st.cache_data
def load_data(url):
    original_df = pd.read_csv(url, dtype={"movieCd": str})
    total_movies = len(original_df)

    required_columns = [
        "movieNm",
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in original_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    usable_df = original_df.copy()

    numeric_columns = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    # Convert numeric fields safely. Invalid text becomes NaN.
    for col in numeric_columns:
        usable_df[col] = pd.to_numeric(
            usable_df[col],
            errors="coerce"
        )

    # Remove rows that cannot produce all four attributes.
    usable_df = usable_df.dropna(
        subset=required_columns
    ).copy()

    usable_df = usable_df[
        (usable_df["first_scrn"] > 0)
        & (usable_df["total_audi"] > 0)
        & (usable_df["first_week_audi"] > 0)
        & (usable_df["days_in_top10"] >= 0)
    ].copy()

    # Original-unit attributes.
    usable_df["Number of Screens"] = usable_df["first_scrn"]
    usable_df["Cumulative Audience"] = usable_df["total_audi"]
    usable_df["Days in Top 10"] = usable_df["days_in_top10"]

    usable_df["Long Run Index"] = (
        usable_df["total_audi"] / usable_df["first_week_audi"]
    ).clip(upper=20)

    # Logarithmic attributes using NumPy.
    # This uses NumPy's vectorized logarithm calculation.
    usable_df["Number of Screens (Log)"] = np.log10(
        usable_df["first_scrn"].where(
            usable_df["first_scrn"] > 0
        )
    )

    usable_df["Cumulative Audience (Log)"] = np.log10(
        usable_df["total_audi"].where(
            usable_df["total_audi"] > 0
        )
    )

    # Keep the original movie order where possible.
    if "movieCd" in usable_df.columns:
        usable_df = usable_df.sort_values(
            "movieCd",
            kind="stable"
        )

    usable_df = usable_df.reset_index(drop=True)

    return original_df, usable_df, total_movies


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
try:
    original_df, df, total_movies = load_data(DATA_URL)

except Exception as error:
    st.error(f"Unable to load movie data: {error}")
    st.stop()

if len(df) < 7:
    st.error(
        "There are not enough valid movie records. "
        "At least 7 usable movies are required."
    )
    st.stop()

st.info(
    f"Using **{len(df):,} movies** out of "
    f"**{total_movies:,} original records**. "
    f"Excluded records: **{total_movies - len(df):,}**."
)

# --------------------------------------------------
# CLUSTERING SETTINGS
# --------------------------------------------------
st.divider()
st.header("⚙️ Clustering Settings")

selected_attributes = st.multiselect(
    "Select attributes for grouping (choose at least two)",
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
    step=1,
    help="Choose how many groups K-means should create."
)

if len(df) < k:
    st.error(
        f"You selected {k} groups, but only {len(df)} usable movies exist."
    )
    st.stop()

# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------
X = df[selected_attributes].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
)

cluster_numbers = model.fit_predict(X_scaled)

# Sort groups by average cumulative audience, highest first.
cluster_order = (
    df.assign(_cluster=cluster_numbers)
    .groupby("_cluster")["Cumulative Audience"]
    .mean()
    .sort_values(ascending=False)
    .index
)

cluster_symbols = {
    cluster_number: SYMBOLS[position]
    for position, cluster_number in enumerate(cluster_order)
}

df["Bundle"] = [
    cluster_symbols[cluster_number]
    for cluster_number in cluster_numbers
]

active_symbols = SYMBOLS[:k]

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
st.divider()
st.header("📊 Movie Group Overview")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Usable Movies", f"{len(df):,}")
metric2.metric("Number of Groups", k)
metric3.metric(
    "Selected Attributes",
    f"{len(selected_attributes)} / 4"
)

# --------------------------------------------------
# 2D SCATTER PLOT
# --------------------------------------------------
st.divider()
st.header("🗺️ Bundle Map · 2D")

axis_col1, axis_col2 = st.columns(2)

x_axis = axis_col1.selectbox(
    "Horizontal Axis",
    options=ATTRIBUTE_LABELS,
    index=0
)

y_axis = axis_col2.selectbox(
    "Vertical Axis",
    options=ATTRIBUTE_LABELS,
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
    template="plotly_dark",
    legend_title="Movie Group"
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Each dot represents one movie. Change the axes to explore "
    "which attributes separate the groups and which attributes overlap."
)

# --------------------------------------------------
# 3D SCATTER PLOT
# --------------------------------------------------
st.divider()
st.header("🌐 Bundle Map · 3D")

if len(selected_attributes) < 3:
    st.info(
        "Select at least three attributes in Clustering Settings "
        "to display the 3D map."
    )
else:
    axis3_col1, axis3_col2, axis3_col3 = st.columns(3)

    x3 = axis3_col1.selectbox(
        "X Axis",
        options=selected_attributes,
        index=0,
        key="x3"
    )

    y3 = axis3_col2.selectbox(
        "Y Axis",
        options=selected_attributes,
        index=1,
        key="y3"
    )

    z3 = axis3_col3.selectbox(
        "Z Axis",
        options=selected_attributes,
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
            "Drag to rotate the 3D map. Hover over a point "
            "to see its movie title."
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
    "Statistics use original units, not logarithmic values. "
    "The Long Run Index is cumulative audience divided by "
    "first-week audience, capped at 20."
)

# --------------------------------------------------
# TOP 5 MOVIES IN EACH GROUP
# --------------------------------------------------
st.divider()
st.header("🏆 Top 5 Movies in Each Group")

group_columns = st.columns(k)

for position, symbol in enumerate(active_symbols):
    group = (
        df[df["Bundle"] == symbol]
        .sort_values(
            "Cumulative Audience",
            ascending=False
        )
        .head(5)
    )

    group_count = int((df["Bundle"] == symbol).sum())

    with group_columns[position]:
        st.subheader(f"{symbol} Group")
        st.caption(f"{group_count:,} movies")

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
# DOWNLOAD RESULTS
# --------------------------------------------------
st.divider()
st.header("📥 Download Results")

csv_data = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="Download Movie Group Data (CSV)",
    data=csv_data,
    file_name="movie_types.csv",
    mime="text/csv"
)

st.success("Movie clustering completed successfully!")
