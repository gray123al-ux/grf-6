import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🎬 Sort Movie Types",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

st.title("🎬 Sort Movie Types")
st.caption(
    "Discover movie types using data, feature selection, "
    "and K-means clustering."
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data(show_spinner="Loading movie data...")
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("Untitled movie").astype(str)

    # Require all four source measurements and a nonzero first-week audience.
    df = df.dropna(subset=numeric_columns).copy()
    df = df[df["first_week_audi"] > 0].copy()

    # Logarithms require positive screen and audience counts.
    df = df[
        (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    ].copy()

    # Four clustering attributes.
    df["log_first_scrn"] = df["first_scrn"].apply(
        lambda x: __import__("math").log10(x)
    )
    df["log_total_audi"] = df["total_audi"].apply(
        lambda x: __import__("math").log10(x)
    )
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    feature_columns = [
        "log_first_scrn",
        "log_total_audi",
        "days_in_top10",
        "long_run_index",
    ]

    # Guard against any invalid values produced by the calculations.
    df = df.dropna(subset=feature_columns)
    df = df[
        df[feature_columns].apply(
            lambda col: col.map(lambda x: abs(x) != float("inf"))
        ).all(axis=1)
    ].copy()

    return df.reset_index(drop=True)


try:
    raw_df = load_data()
except Exception as e:
    st.error("The movie dataset could not be loaded.")
    st.exception(e)
    st.info(
        "Check the GitHub CSV URL, column names, and network access, "
        "then refresh the app."
    )
    st.stop()

if len(raw_df) < 3:
    st.error("At least three valid movies are needed for three clusters.")
    st.stop()

# --------------------------------------------------
# Feature definitions
# --------------------------------------------------
FEATURES = {
    "Number of screens (log₁₀)": {
        "column": "log_first_scrn",
        "description": "Common logarithm of the number of screens",
        "original": "first_scrn",
    },
    "Cumulative audience (log₁₀)": {
        "column": "log_total_audi",
        "description": "Common logarithm of cumulative audience",
        "original": "total_audi",
    },
    "Days in Top 10": {
        "column": "days_in_top10",
        "description": "Number of days ranked in the daily Top 10",
        "original": "days_in_top10",
    },
    "Long-run index": {
        "column": "long_run_index",
        "description": "Cumulative audience ÷ first-week audience, capped at 20",
        "original": "long_run_index",
    },
}

ORIGINAL_LABELS = {
    "first_scrn": "Screens",
    "total_audi": "Cumulative audience",
    "days_in_top10": "Days in Top 10",
    "long_run_index": "Long-run index",
}

GROUP_LABELS = ["㉮", "㉯", "㉰"]
GROUP_COLORS = {
    "㉮": "#EF553B",
    "㉯": "#636EFA",
    "㉰": "#00CC96",
}

# --------------------------------------------------
# Feature selection
# --------------------------------------------------
st.header("1. Choose clustering attributes")

st.write(
    "Select at least two attributes. The selected attributes are "
    "standardized before K-means clustering."
)

selected_features = st.multiselect(
    "Attributes used to group movies",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    help="Choose two, three, or all four attributes.",
)

if len(selected_features) < 2:
    st.warning("Please select at least two attributes to continue.")
    st.stop()

selected_columns = [
    FEATURES[name]["column"] for name in selected_features
]

with st.expander("What do these attributes mean?"):
    for name in selected_features:
        st.markdown(
            f"- **{name}:** {FEATURES[name]['description']}"
        )

# --------------------------------------------------
# Standardization and K-means
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(raw_df[selected_columns])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

raw_df["cluster_id"] = kmeans.fit_predict(X_scaled)

# Give ㉮ to the cluster with the highest average cumulative audience.
cluster_order = (
    raw_df.groupby("cluster_id")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster_id: GROUP_LABELS[position]
    for position, cluster_id in enumerate(cluster_order)
}

raw_df["Group"] = raw_df["cluster_id"].map(label_map)

# --------------------------------------------------
# Dataset summary
# --------------------------------------------------
st.header("2. Dataset and grouping results")

total_valid = len(raw_df)
group_counts = raw_df["Group"].value_counts()

summary_text = (
    f"**Valid movies: {total_valid:,}**　|　"
    f"**Grouped into 3 types:** "
    + " · ".join(
        f"{group} {int(group_counts.get(group, 0)):,} movies"
        for group in GROUP_LABELS
    )
)

st.markdown(summary_text)

metric_cols = st.columns(4)
metric_cols[0].metric("Valid movies", f"{total_valid:,}")
metric_cols[1].metric("㉮ movies", f"{int(group_counts.get('㉮', 0)):,}")
metric_cols[2].metric("㉯ movies", f"{int(group_counts.get('㉯', 0)):,}")
metric_cols[3].metric("㉰ movies", f"{int(group_counts.get('㉰', 0)):,}")

st.caption(
    "Movies with missing required measurements, a first-week audience "
    "of zero or less, or nonpositive screen/cumulative-audience counts "
    "are excluded. The ㉮–㉰ labels are assigned by descending average "
    "cumulative audience, not by K-means' arbitrary internal IDs."
)

# --------------------------------------------------
# 2D scatter plot
# --------------------------------------------------
st.header("3. Explore the groups in 2D")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "Horizontal axis (X)",
        options=selected_features,
        index=0,
    )

with axis_col2:
    default_y = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "Vertical axis (Y)",
        options=selected_features,
        index=default_y,
    )

if x_feature == y_feature:
    st.warning(
        "The horizontal and vertical axes use the same attribute. "
        "Choose different attributes to compare two dimensions."
    )
else:
    x_col = FEATURES[x_feature]["column"]
    y_col = FEATURES[y_feature]["column"]

    fig2d = px.scatter(
        raw_df,
        x=x_col,
        y=y_col,
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
            x_col: ":.3f",
            y_col: ":.3f",
            "cluster_id": False,
        },
        labels={
            x_col: x_feature,
            y_col: y_feature,
            "Group": "Movie type",
        },
        title=f"Movie types: {x_feature} vs {y_feature}",
    )

    fig2d.update_traces(
        marker={"size": 9, "opacity": 0.78},
    )
    fig2d.update_layout(
        height=580,
        legend_title_text="Movie type",
        hoverlabel={"namelength": -1},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )

    st.plotly_chart(fig2d, use_container_width=True)

st.caption(
    "Each point represents one movie. Hover over a point to see its title "
    "and original measurements. Nearby points have similar values on "
    "the two selected axes."
)

# --------------------------------------------------
# 3D scatter plot
# --------------------------------------------------
st.header("4. Explore the groups in 3D")

if len(selected_features) < 3:
    st.info(
        "Select at least three attributes above to display the 3D scatter "
        "plot. Your current grouping results and 2D chart will still work."
    )
else:
    three_d_cols = st.columns(3)

    with three_d_cols[0]:
        x3_feature = st.selectbox(
            "3D X axis",
            options=selected_features,
            index=0,
            key="x3",
        )

    with three_d_cols[1]:
        y3_feature = st.selectbox(
            "3D Y axis",
            options=selected_features,
            index=1,
            key="y3",
        )

    with three_d_cols[2]:
        z3_feature = st.selectbox(
            "3D Z axis",
            options=selected_features,
            index=2,
            key="z3",
        )

    if len({x3_feature, y3_feature, z3_feature}) < 3:
        st.warning("Choose three different attributes for the 3D axes.")
    else:
        x3_col = FEATURES[x3_feature]["column"]
        y3_col = FEATURES[y3_feature]["column"]
        z3_col = FEATURES[z3_feature]["column"]

        fig3d = px.scatter_3d(
            raw_df,
            x=x3_col,
            y=y3_col,
            z=z3_col,
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
                x3_col: x3_feature,
                y3_col: y3_feature,
                z3_col: z3_feature,
                "Group": "Movie type",
            },
            title="Three-dimensional movie type comparison",
        )

        fig3d.update_traces(
            marker={"size": 3, "opacity": 0.8},
        )
        fig3d.update_layout(
            height=700,
            legend_title_text="Movie type",
            hoverlabel={"namelength": -1},
            margin={"l": 0, "r": 0, "t": 70, "b": 0},
            scene={
                "aspectmode": "cube",
                "xaxis": {"title": x3_feature},
                "yaxis": {"title": y3_feature},
                "zaxis": {"title": z3_feature},
            },
        )

        st.plotly_chart(fig3d, use_container_width=True)

        st.caption(
            "Drag to rotate the 3D view and use the scroll wheel or "
            "trackpad to zoom. The plot uses a cube-shaped 3D scene."
        )

# --------------------------------------------------
# Group summary: original measurement units
# --------------------------------------------------
st.header("5. Compare the three movie types")

summary_rows = []

for group in GROUP_LABELS:
    group_df = raw_df[raw_df["Group"] == group]

    summary_rows.append(
        {
            "Group": group,
            "Number of movies": len(group_df),
            "Average screens": group_df["first_scrn"].mean(),
            "Average cumulative audience": group_df["total_audi"].mean(),
            "Average days in Top 10": group_df["days_in_top10"].mean(),
            "Average long-run index": group_df["long_run_index"].mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)

display_summary = summary_df.copy()
display_summary["Average screens"] = display_summary[
    "Average screens"
].round(1)
display_summary["Average cumulative audience"] = display_summary[
    "Average cumulative audience"
].round(1)
display_summary["Average days in Top 10"] = display_summary[
    "Average days in Top 10"
].round(2)
display_summary["Average long-run index"] = display_summary[
    "Average long-run index"
].round(2)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Group": st.column_config.TextColumn("Movie type"),
        "Number of movies": st.column_config.NumberColumn(
            "Number of movies", format="%d"
        ),
        "Average screens": st.column_config.NumberColumn(
            "Average screens", format="%,.1f"
        ),
        "Average cumulative audience": st.column_config.NumberColumn(
            "Average cumulative audience", format="%,.1f"
        ),
        "Average days in Top 10": st.column_config.NumberColumn(
            "Average days in Top 10", format="%.2f"
        ),
        "Average long-run index": st.column_config.NumberColumn(
            "Average long-run index", format="%.2f"
        ),
    },
)

st.caption(
    "The table shows averages in the original units, not standardized "
    "values or logarithms. The long-run index is capped at 20 for each "
    "movie before calculating its group average."
)

# --------------------------------------------------
# Top five movies by cumulative audience
# --------------------------------------------------
st.header("6. Top five movies in each type")

st.write(
    "Movies are ranked by cumulative audience within each group. "
    "If a group contains fewer than five movies, all available titles "
    "are shown."
)

group_tabs = st.tabs(
    [f"{group} · {int(group_counts.get(group, 0))} movies" for group in GROUP_LABELS]
)

for tab, group in zip(group_tabs, GROUP_LABELS):
    with tab:
        top_movies = (
            raw_df[raw_df["Group"] == group]
            .sort_values("total_audi", ascending=False)
            .head(5)
            .copy()
        )

        if top_movies.empty:
            st.info("There are no movies in this group.")
        else:
            top_movies = top_movies.reset_index(drop=True)
            top_movies.insert(
                0,
                "Rank",
                range(1, len(top_movies) + 1),
            )

            top_movies = top_movies[
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
                top_movies,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank", format="%d"
                    ),
                    "Screens": st.column_config.NumberColumn(
                        "Screens", format="%d"
                    ),
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

# --------------------------------------------------
# Method and interpretation
# --------------------------------------------------
with st.expander("How does this clustering work?"):
    st.markdown(
        """
        1. **Prepare the data:** Exclude movies with missing required
           measurements, zero or negative first-week audience, or
           nonpositive screen/cumulative-audience counts.
        2. **Transform the scale:** Apply log₁₀ to screen counts and
           cumulative audience. Keep Top 10 days unchanged and calculate
           the long-run index as cumulative audience divided by first-week
           audience, capped at 20.
        3. **Choose attributes:** Select at least two attributes.
        4. **Standardize:** Scale selected attributes to a comparable
           statistical scale.
        5. **Cluster:** K-means separates the movies into three groups.
           The fixed random seed and `n_init=10` make the result
           reproducible for the same data and software environment.
        6. **Name the groups:** ㉮ has the highest average cumulative
           audience, followed by ㉯ and ㉰.

        **Important:** These are exploratory groups, not objectively
        correct movie categories. Changing the selected attributes can
        change the clusters. The group label is assigned after clustering
        and does not influence how K-means forms the groups.
        """
    )

st.divider()
st.caption(
    "Sort Movie Types · Data source: KOBIS movie dataset · "
    "Built with Streamlit, pandas, Plotly, and scikit-learn."
)
