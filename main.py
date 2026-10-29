import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Sort Movie Types",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

FEATURES = {
    "Number of screens (log₁₀)": "log_first_scrn",
    "Cumulative audience (log₁₀)": "log_total_audi",
    "Days in Top 10": "days_in_top10",
    "Long-run index": "long_run_index",
}

GROUP_LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
GROUP_COLORS = {
    "㉮": "#EF553B",
    "㉯": "#636EFA",
    "㉰": "#00CC96",
    "㉱": "#AB63FA",
    "㉲": "#FFA15A",
    "㉳": "#19D3F3",
    "㉴": "#FF6692",
}

st.title("🎬 Sort Movie Types")
st.caption(
    "Group similar movies using box-office data and K-means clustering. "
    "There is no predefined correct-answer category."
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load and prepare the data
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading movie data...")
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")

    required = [
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required column(s): " + ", ".join(missing)
        )

    numeric = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # A movie must have all four source values, and first-week audience
    # must be greater than zero to calculate the long-run index.
    df = df.dropna(subset=numeric).copy()
    df = df[df["first_week_audi"] > 0].copy()

    # Logarithms are defined here only for positive counts.
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)].copy()

    df["log_first_scrn"] = df["first_scrn"].map(math.log10)
    df["log_total_audi"] = df["total_audi"].map(math.log10)
    
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    feature_columns = list(FEATURES.values())
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_columns).copy()
    df["movieNm"] = df["movieNm"].fillna("Untitled movie").astype(str)

    return df.reset_index(drop=True)


try:
    movies = load_data(DATA_URL)
except Exception as exc:
    st.error("Could not load the movie dataset.")
    st.exception(exc)
    st.info(
        "Check the CSV URL, column names, and network connection, then "
        "refresh the app."
    )
    st.stop()

if len(movies) < 3:
    st.error("At least three valid movies are required for three clusters.")
    st.stop()


# ------------------------------------------------------------
# Select attributes
# ------------------------------------------------------------
st.header("1. Choose attributes for grouping")

selected_features = st.multiselect(
    "Clustering attributes (select at least two)",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    help=(
        "Screen counts and cumulative audience use log₁₀. "
        "Days in Top 10 stays unchanged. The long-run index is capped at 20."
    ),
)

if len(selected_features) < 2:
    st.warning("Select at least two attributes to continue.")
    st.stop()

selected_columns = [FEATURES[name] for name in selected_features]

with st.expander("What do the attributes mean?"):
    st.markdown(
        "- **Number of screens (log₁₀):** Common logarithm of screens at release.\n"
        "- **Cumulative audience (log₁₀):** Common logarithm of total audience.\n"
        "- **Days in Top 10:** Number of days the movie appeared in the daily Top 10.\n"
        "- **Long-run index:** Cumulative audience ÷ first-week audience, capped at 20."
    )


# ------------------------------------------------------------
# Cluster movies
# ------------------------------------------------------------
scaler = StandardScaler()
scaled_values = scaler.fit_transform(movies[selected_columns])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
movies["cluster_id"] = kmeans.fit_predict(scaled_values)

# The labels are assigned after clustering: ㉮ has the highest mean
# cumulative audience, then ㉯, then ㉰.
cluster_order = (
    movies.groupby("cluster_id")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_to_label = {
    cluster_id: GROUP_LABELS[index]
    for index, cluster_id in enumerate(cluster_order)
}
movies["Group"] = movies["cluster_id"].map(cluster_to_label)

group_counts = movies["Group"].value_counts()
st.header("2. Dataset and grouping results")

counts_line = "　|　".join(
    f"{label}: {int(group_counts.get(label, 0)):,} movies"
    for label in GROUP_LABELS[:3]
)
st.markdown(
    f"**Valid movies: {len(movies):,}**　|　**Groups:** {counts_line}"
)

metric_columns = st.columns(4)
metric_columns[0].metric("Valid movies", f"{len(movies):,}")
for index, label in enumerate(GROUP_LABELS[:3], start=1):
     metric_columns[index].metric(
        f"{label} movies",
        f"{int(group_counts.get(label, 0)):,}",
    )

st.caption(
    "Rows with missing required values, a first-week audience of zero or less, "
    "or nonpositive screen/cumulative-audience counts are excluded. "
    "㉮ is assigned to the group with the highest average cumulative audience."
)


# ------------------------------------------------------------
# 2D scatter plot
# ------------------------------------------------------------
st.header("3. Explore the groups in 2D")
axis_left, axis_right = st.columns(2)

with axis_left:
    x_feature = st.selectbox(
        "Horizontal axis (X)",
        options=selected_features,
        index=0,
    )
with axis_right:
    y_default = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "Vertical axis (Y)",
        options=selected_features,
        index=y_default,
    )

if x_feature == y_feature:
    st.warning("Choose different attributes for the X and Y axes.")
else:
    x_column = FEATURES[x_feature]
    y_column = FEATURES[y_feature]

    fig2d = px.scatter(
        movies,
        x=x_column,
        y=y_column,
        color="Group",
        color_discrete_map=GROUP_COLORS,
        category_orders={"Group": GROUP_LABELS},
        hover_name="movieNm",
        hover_data={
            "Group": True,
            "first_scrn": ":,",
            "first_week_audi": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "long_run_index": ":.2f",
            x_column: ":.3f",
            y_column: ":.3f",
            "cluster_id": False,
        },
        labels={
            x_column: x_feature,
            y_column: y_feature,
            "Group": "Movie type",
        },
        title=f"{x_feature} vs {y_feature}",
    )
    fig2d.update_traces(marker={"size": 8, "opacity": 0.8})
    fig2d.update_layout(
        height=560,
        legend_title_text="Movie type",
        hoverlabel={"namelength": -1},
        margin={"l": 20, "r": 20, "t": 65, "b": 20},
    )
    st.plotly_chart(fig2d, use_container_width=True)

st.caption(
    "Each dot is a movie. Hover over a dot to see the movie title and "
    "its measurements in the original dataset."
)


# ------------------------------------------------------------
# 3D scatter plot
# ------------------------------------------------------------
st.header("4. Explore the groups in 3D")

if len(selected_features) < 3:
    st.info(
        "Select at least three clustering attributes above to display the "
        "3D scatter plot. The 2D plot and group summary remain available."
    )
else:
    three_d_columns = st.columns(3)
    with three_d_columns[0]:
        x3_feature = st.selectbox(
            "3D X axis",
            options=selected_features,
            index=0,
            key="x3_feature",
        )
    with three_d_columns[1]:
        y3_feature = st.selectbox(
            
            "3D Y axis",
            options=selected_features,
            index=1,
            key="y3_feature",
        )
    with three_d_columns[2]:
        z3_feature = st.selectbox(
            "3D Z axis",
            options=selected_features,
            index=2,
            key="z3_feature",
        )

    if len({x3_feature, y3_feature, z3_feature}) != 3:
        st.warning("Choose three different attributes for the 3D axes.")
    else:
        x3_column = FEATURES[x3_feature]
        y3_column = FEATURES[y3_feature]
        z3_column = FEATURES[z3_feature]

        fig3d = px.scatter_3d(
            movies,
            x=x3_column,
            y=y3_column,
            z=z3_column,
            color="Group",
            color_discrete_map=GROUP_COLORS,
            category_orders={"Group": GROUP_LABELS},
            hover_name="movieNm",
            hover_data={
                "Group": True,
                "first_scrn": ":,",
                "first_week_audi": ":,",
                "total_audi": ":,",
                "days_in_top10": True,
                "long_run_index": ":.2f",
                "cluster_id": False,
            },
            labels={
                x3_column: x3_feature,
                y3_column: y3_feature,
                z3_column: z3_feature,
                "Group": "Movie type",
            },
            title="Three-dimensional movie type comparison",
        )
        fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
        fig3d.update_layout(
            height=680,
            legend_title_text="Movie type",
            hoverlabel={"namelength": -1},
            margin={"l": 0, "r": 0, "t": 65, "b": 0},
            scene={
                "aspectmode": "cube",
                "xaxis": {"title": x3_feature},
                "yaxis": {"title": y3_feature},
                "zaxis": {"title": z3_feature},
            },
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            "Drag the plot to rotate it and use the scroll wheel or trackpad "
            "to zoom. The 3D scene uses a cube-shaped aspect ratio."
        )


# ------------------------------------------------------------
# Group averages in original units
# ------------------------------------------------------------
st.header("5. Compare the three movie types")

summary_rows = []
for label in GROUP_LABELS[:3]:
    group_movies = movies[movies["Group"] == label]
    summary_rows.append(
        {
            "Group": label,
            "Number of movies": len(group_movies),
            "Average screens": group_movies["first_scrn"].mean(),
            "Average cumulative audience": group_movies["total_audi"].mean(),
            "Average days in Top 10": group_movies["days_in_top10"].mean(),
            "Average long-run index": group_movies["long_run_index"].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)
summary["Average screens"] = summary["Average screens"].round(1)
summary["Average cumulative audience"] = (
    summary["Average cumulative audience"].round(1)
)
summary["Average days in Top 10"] = summary["Average days in Top 10"].round(2)
summary["Average long-run index"] = summary["Average long-run index"].round(2)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Group": st.column_config.TextColumn("Movie type"),
        "Number of movies": st.column_config.NumberColumn(
            "Number of movies", format="%d"
        ),
st.caption(
    "Averages are shown in original units, not standardized values or "
    "logarithms. The long-run index is capped at 20 per movie."
)


# ------------------------------------------------------------
# Top five movies per group
# ------------------------------------------------------------
st.header("6. Top five movies in each group")
st.write(
    "The lists are sorted by cumulative audience, highest first. "
    "If a group has fewer than five movies, all its movies are shown."
)

tabs = st.tabs(
    [
        f"{label} · {int(group_counts.get(label, 0))} movies"
        for label in GROUP_LABELS[:3]
    ]
)

for tab, label in zip(tabs, GROUP_LABELS[:3]):
    with tab:
        top = (
            movies[movies["Group"] == label]
            .sort_values("total_audi", ascending=False)
            .head(5)
            .copy()
            .reset_index(drop=True)
        )

        if top.empty:
            st.info("There are no movies in this group.")
        else:
            top.insert(0, "Rank", range(1, len(top) + 1))
            top = top[
                [
                    "Rank",
                    "movieNm",
                    "first_scrn",
                    "first_week_audi",
                    "total_audi",
                    "days_in_top10",
                    "long_run_index",
                ]
            ].rename(
                columns={
                    "movieNm": "Movie title",
                    "first_scrn": "Screens",
                    "first_week_audi": "First-week audience",
                    "total_audi": "Cumulative audience",
                    "days_in_top10": "Days in Top 10",
                    "long_run_index": "Long-run index",
                }
            )
            st.dataframe(
                top,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Screens": st.column_config.NumberColumn("Screens", format="%d"),
                    "First-week audience": st.column_config.NumberColumn(
                        "First-week audience", format="%d"
                    ),
                    "Cumulative audience": st.column_config.NumberColumn(
                        "Cumulative audience", format="%d"
                    ),
                    "Days in Top 10": st.column_config.NumberColumn(
                        "Days in Top 10", format="%d"
                    ),
                    "Long-run index": st.column_config.NumberColumn(
                        "Long-run index", format="%.2f"
                    ),
                },
            )


# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------
with st.expander("How the grouping works"):
    st.markdown(
        """
        1. **Prepare the data:** Exclude rows missing any required source
           attribute, rows with first-week audience of zero or less, and
           rows with nonpositive screen or cumulative-audience counts.
        2. **Transform the data:** Apply log₁₀ to screen counts and
           cumulative audience. Keep Top 10 days unchanged. Calculate the
           long-run index as cumulative audience divided by first-week
           audience, capped at 20.
        3. **Choose attributes:** Select at least two attributes.
        4. **Standardize:** Convert the selected attributes to a comparable
           scale using `StandardScaler`.
        5. **Cluster:** K-means groups movies into three clusters using
           `random_state=42` and `n_init=10`.
        6. **Assign labels:** ㉮ has the highest average cumulative audience,
           ㉯ the next highest, and ㉰ the lowest.

        Clustering is unsupervised learning: there is no predefined correct
        answer. The results depend on the selected attributes and the data.
        """
    )

st.divider()
st.caption(
    "Sort Movie Types · Data source: KOBIS movie dataset · "
    "Built with Streamlit, pandas, Plotly, and scikit-learn."
)
