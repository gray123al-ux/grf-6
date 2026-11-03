
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Sort Movie Types",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

REQUIRED_COLUMNS = [
    "movieCd", "movieNm", "openDt", "genre", "nation",
    "first_scrn", "first_show", "first_date", "peak",
    "first_week_audi", "total_audi", "days_in_top10"
]

FEATURE_NAMES = {
    "log_first_scrn": "Opening Screens (log)",
    "log_total_audi": "Total Audience (log)",
    "days_in_top10": "Days in Top 10",
    "long_run_index": "Long-run Index"
}

LABEL_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]


# ==================================================
# CUSTOM STYLE
# ==================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        margin-bottom: 25px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        padding: 16px;
        border-radius: 12px;
    }
    .section-note {
        padding: 12px 16px;
        border-radius: 10px;
        background-color: rgba(100,140,200,0.10);
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">🎬 Sort Movie Types</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Discover movie patterns with K-Means clustering'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data(show_spinner="Loading movie data...")
def load_data(url):

    df = pd.read_csv(url, encoding="utf-8-sig")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    total_movies = len(df)

    numeric_columns = [
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Logarithmic features
    df["log_first_scrn"] = np.log10(
        df["first_scrn"].where(df["first_scrn"] > 0)
    )

    df["log_total_audi"] = np.log10(
        df["total_audi"].where(df["total_audi"] > 0)
    )

    # Long-run index, capped at 20
    df["long_run_index"] = (
        df["total_audi"] /
        df["first_week_audi"].where(
            df["first_week_audi"] > 0
        )
    ).clip(upper=20)

    # Remove missing and invalid observations
    feature_columns = [
        "log_first_scrn",
        "log_total_audi",
        "days_in_top10",
        "long_run_index"
    ]

    usable_df = df.dropna(
        subset=feature_columns + ["movieNm"]
    ).copy()

    usable_df = usable_df[
        (usable_df["first_scrn"] > 0) &
        (usable_df["total_audi"] > 0) &
        (usable_df["first_week_audi"] > 0) &
        (usable_df["days_in_top10"] >= 0)
    ].copy()

    # Remove infinite values
    usable_df = usable_df[
        np.isfinite(usable_df[feature_columns]).all(axis=1)
    ].copy()

    return df, usable_df, total_movies


# ==================================================
# LOAD DATA WITH ERROR HANDLING
# ==================================================

try:
    original_df, usable_df, total_movies = load_data(DATA_URL)

except Exception as error:
    st.error("Could not load or process the movie data.")
    st.write("Check the CSV URL, internet connection, and column names.")
    st.exception(error)
    st.stop()


if len(usable_df) < 3:
    st.error("At least three valid movies are required.")
    st.stop()


# ==================================================
# DATA OVERVIEW
# ==================================================

st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Movies", f"{total_movies:,}")
c2.metric("Usable Movies", f"{len(usable_df):,}")
c3.metric(
    "Excluded Movies",
    f"{total_movies - len(usable_df):,}"
)

st.caption(
    "Movies with missing or invalid clustering values are excluded."
)


# ==================================================
# FEATURE SELECTION
# ==================================================

st.divider()
st.header("1. Choose Your Features")

st.write(
    "Choose the movie characteristics that K-Means will use "
    "to create groups."
)

selected_features = st.multiselect(
    "Select at least two attributes:",
    options=list(FEATURE_NAMES.keys()),
    default=list(FEATURE_NAMES.keys()),
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()


# ==================================================
# PREPARE DATA AND CHECK CLUSTER LIMIT
# ==================================================

X = usable_df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

unique_rows = np.unique(X_scaled, axis=0).shape[0]

max_clusters = min(
    7,
    len(usable_df),
    unique_rows
)

if max_clusters < 2:
    st.error("There are not enough distinct movie records to create groups.")
    st.stop()


# ==================================================
# ELBOW METHOD
# ==================================================

st.divider()
st.header("2. Find a Suitable Number of Groups")

st.write(
    "The Elbow Method compares the within-group sum of squared "
    "distances (inertia) for different group counts. "
    "Look for a point where the decrease begins to slow down."
)

elbow_values = []

for k in range(1, max_clusters + 1):

    elbow_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    elbow_model.fit(X_scaled)

    elbow_values.append({
        "Number of Groups": k,
        "Inertia": elbow_model.inertia_
    })

elbow_df = pd.DataFrame(elbow_values)

fig_elbow = px.line(
    elbow_df,
    x="Number of Groups",
    y="Inertia",
    markers=True,
    title="Elbow Method"
)

fig_elbow.update_layout(height=420)

st.plotly_chart(
    fig_elbow,
    use_container_width=True
)

# Calculate reduction from previous group count
elbow_df["Reduction"] = elbow_df["Inertia"].shift(1) - elbow_df["Inertia"]

elbow_df["Reduction (%)"] = (
    elbow_df["Reduction"] /
    elbow_df["Inertia"].shift(1) * 100
)

st.dataframe(
    elbow_df.style.format({
        "Inertia": "{:,.2f}",
        "Reduction": "{:,.2f}",
        "Reduction (%)": "{:.1f}%"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "A larger reduction means adding another group improved "
    "the clustering more. The elbow is a guide, not an automatic answer."
)


# ==================================================
# CLUSTER COUNT
# ==================================================

st.header("3. Choose the Number of Movie Groups")

n_clusters = st.slider(
    "Number of groups",
    min_value=2,
    max_value=max_clusters,
    value=min(3, max_clusters),
    help="Choose a group count based on the Elbow Method and your analysis."
)


# ==================================================
# K-MEANS CLUSTERING
# ==================================================

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

usable_df = usable_df.copy()
usable_df["cluster"] = model.fit_predict(X_scaled)

# Sort groups by average total audience
cluster_order = (
    usable_df.groupby("cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_mapping = {
    cluster: LABEL_SYMBOLS[index]
    for index, cluster in enumerate(cluster_order)
}

usable_df["group"] = usable_df["cluster"].map(cluster_mapping)

group_order = [
    cluster_mapping[cluster]
    for cluster in cluster_order
]


# ==================================================
# CLUSTER OVERVIEW
# ==================================================

st.divider()
st.header("4. Clustering Results")

st.success(
    f"Created {n_clusters} movie groups from "
    f"{len(usable_df):,} usable movies."
)

st.caption(
    "Group labels are ordered by average total audience. "
    "Group ㉮ has the highest average total audience."
)


# ==================================================
# 2D SCATTER PLOT
# ==================================================

st.divider()
st.header("5. Two-Dimensional Movie Map")

left, right = st.columns(2)

with left:
    x_feature = st.selectbox(
        "Horizontal axis",
        selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
        key="x_feature"
    )

with right:
    y_default = 1 if selected_features[1] != x_feature else 0

    y_feature = st.selectbox(
        "Vertical axis",
        selected_features,
        index=y_default,
        format_func=lambda x: FEATURE_NAMES[x],
        key="y_feature"
    )

if x_feature == y_feature:
    st.warning(
        "Choose different features for the horizontal and vertical axes."
    )
else:
    fig_2d = px.scatter(
        usable_df,
        x=x_feature,
        y=y_feature,
        color="group",
        hover_name="movieNm",
        hover_data={
            "first_scrn": ":,",
            "first_week_audi": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "long_run_index": ":.2f"
        },
        category_orders={"group": group_order},
        labels={
            x_feature: FEATURE_NAMES[x_feature],
            y_feature: FEATURE_NAMES[y_feature],
            "group": "Movie Group"
        },
        title="Movie Groups by Selected Features"
    )

    fig_2d.update_traces(
        marker=dict(size=9, opacity=0.8)
    )

    fig_2d.update_layout(height=600)

    st.plotly_chart(
        fig_2d,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-note">'
        '💡 Movies close together have similar values '
        'for the selected features.'
        '</div>',
        unsafe_allow_html=True
    )


# ==================================================
# 3D SCATTER PLOT
# ==================================================

st.divider()
st.header("6. Three-Dimensional Movie Map")

if len(selected_features) >= 3:

    col1, col2, col3 = st.columns(3)

    with col1:
        z_x = st.selectbox(
            "3D X axis",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x],
            key="3d_x"
        )

    with col2:
        z_y = st.selectbox(
            "3D Y axis",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_NAMES[x],
            key="3d_y"
        )

    with col3:
        z_z = st.selectbox(
            "3D Z axis",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_NAMES[x],
            key="3d_z"
        )

    if len({z_x, z_y, z_z}) < 3:
        st.warning("Choose three different features for the 3D graph.")
    else:
        fig_3d = px.scatter_3d(
            usable_df,
            x=z_x,
            y=z_y,
            z=z_z,
            color="group",
            hover_name="movieNm",
            hover_data={
                "total_audi": ":,",
                "first_week_audi": ":,",
                "days_in_top10": True
            },
            category_orders={"group": group_order},
            labels={
                z_x: FEATURE_NAMES[z_x],
                z_y: FEATURE_NAMES[z_y],
                z_z: FEATURE_NAMES[z_z],
                "group": "Movie Group"
            },
            title="3D Movie Group Distribution"
        )

        fig_3d.update_traces(
            marker=dict(size=4, opacity=0.8)
        )

        fig_3d.update_layout(
            height=700,
            scene=dict(aspectmode="cube")
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True
        )

        st.caption("Drag the graph to rotate the 3D view.")

else:
    st.info(
        "Select at least three features to display the 3D graph."
    )


# ==================================================
# GROUP SUMMARY
# ==================================================

st.divider()
st.header("7. Compare Movie Groups")

summary = (
    usable_df.groupby("group")
    .agg(
        Movie_Count=("movieNm", "count"),
        Average_Screens=("first_scrn", "mean"),
        Average_Audience=("total_audi", "mean"),
        Average_Top10_Days=("days_in_top10", "mean"),
        Average_Long_Run=("long_run_index", "mean")
    )
    .reindex(group_order)
    .reset_index()
)

summary.columns = [
    "Group",
    "Movie Count",
    "Average Screens",
    "Average Total Audience",
    "Average Top 10 Days",
    "Average Long-run Index"
]

st.dataframe(
    summary.style.format({
        "Movie Count": "{:,.0f}",
        "Average Screens": "{:,.0f}",
        "Average Total Audience": "{:,.0f}",
        "Average Top 10 Days": "{:.1f}",
        "Average Long-run Index": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# REPRESENTATIVE MOVIES
# ==================================================

st.divider()
st.header("8. Representative Movies")

for group in group_order:

    group_df = (
        usable_df[usable_df["group"] == group]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(
        f"Group {group} — Top 5 Movies",
        expanded=True
    ):

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
                "movieNm": "Movie",
                "genre": "Genre",
                "first_scrn": "Opening Screens",
                "first_week_audi": "First Week Audience",
                "total_audi": "Total Audience",
                "days_in_top10": "Days in Top 10",
                "long_run_index": "Long-run Index"
            }
        )

        st.dataframe(
            display_df.style.format({
                "Opening Screens": "{:,.0f}",
                "First Week Audience": "{:,.0f}",
                "Total Audience": "{:,.0f}",
                "Days in Top 10": "{:.0f}",
                "Long-run Index": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# CONCLUSION
# ==================================================

st.divider()
st.header("💡 Explore the Results")

st.markdown("""
Use these questions to interpret your results:

- What characteristics do movies in Group ㉮ share?
- How are Groups ㉯ and ㉰ different?
- Do movies with more opening screens always attract more viewers?
- Can a movie with a small first-week audience have a high long-run index?
- What happens when you change the selected features?
- Does the Elbow Method suggest the same number of groups you selected?
""")

st.caption(
    "Movie data clustering experiment · K-Means"
)
