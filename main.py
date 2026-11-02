# main.py — Dividing Movie Types: Grouping similar movies without a fixed answer 
import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.express as px 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 

st.set_page_config(page_title="Dividing Movie Types", page_icon="🎬", layout="wide") 
st.title("🎬 Dividing Movie Types") 
st.caption("Grouping movies based on attributes I selected that are similar, without a fixed answer column.") 

MOVIES = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv" 
Symbol = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"] # Display attribute used instead of group number 
= ["Number of screens (log)", "Cumulative [Audience (log)", "Days in Top 10", "Long Run Index"] 
Original Unit = ["Number of Screens", "Cumulative Audience", "Days in Top 10", "Long Run Index"] 


@st.cache_data 
def load_data(): 
    df = pd.read_csv("MOVIES", dtype={"movieCd": str}) 
    df = df.sort_values("movieCd").reset_index(drop=True) 
    Total Number of Films = len(df) 
    # Exclude movies for which four attributes cannot be created: rows with no value or first-week audience of 0 Use 
    Column = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"] 
    df = df.dropna(subset=useColumn + ["movieNm"]) 
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0) & (df["first_week_audi"] > 0)] 
    df = df.reset_index(drop=True) 
    return df, Total_Audio 


df, Total_Audio = load_data() 

# Create four attributes. For two numbers with a very large difference, take the common logarithm 
df["Number of Screens"] = df["first_scrn"] 
df["Cumulative Audience"] = df["total_audi"] 
df["Days in Top 10"] = df["days_in_top10"] 
df["Long Run Index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20) # If it exceeds 20, set it to 20 
df["Number of Screens(Log)"] = np.log10(df["first_scrn"]) 
df["Cumulative Audience(Log)"] = np.log10(df["total_audi"]) 

st.info(f"Grouped into {len(df):,} episodes among the total {total number of episodes:,} episodes that can create all four attributes.") 

EqualAttributes = st.multiselect("Attributes to use for grouping (two or more)", attribute, default=attribute) 
if len(selectedattribute) < 2: 
    st.warning("Please select two or more attributes.") 
    st.stop() 

# Standardize the selected attributes and divide by k-means. Fix the random number so that the same result is obtained even if run again 
k = 3 
Xs = StandardScaler().fit_transform(df[selectedattribute]) 
number = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs) 

# Display in the order of ㉮·㉯·㉰ starting from the group with the largest cumulative audience average 
order = pd.Series(df["cumulative audience"].to_numpy()).groupby(number).mean().sort_values(ascending=False).index 
digit = {g: i for i,g in enumerate(order)} 
df["Bundle"] = [symbol[position[g]] for g in number] 

st.subheader("Bundle Map · 2D") 
c1, c2 = st.columns(2) 
x-axis = c1.selectbox("Horizontal Axis", uniform property, index=0) 
y-axis = c2.selectbox("Vertical Axis", uniform property, index=min(1, len(uniform property) - 1)) 
fig = px.scatter(df, x=x-axis, y=y-axis, color="Bundle", hover_name="movieNm", 
                 category_orders={"Bundle": symbol[:k]})
fig.update_traces(marker=dict(size=9, opacity=0.8)) 
fig.update_layout(height=460) 
st.plotly_chart(fig, width="stretch") 
st.caption("Each point represents a movie. If you change the axis, you can see the axis where the bundles are divided and the axis where they are mixed.") 

st.subheader("Bundled Map · 3D") 
if len(EqualAttribute) < 3: 
    st.info("Select three or more attributes to display the 3D plot.") 
else: 
    d1, d2, d3 = st.columns(3) 
    x3 = d1.selectbox("x-axis", equalAttribute, index=0, key="x3") 
    y3 = d2.selectbox("y-axis", equalAttribute, index=1, key="y3") 
    z3 = d3.selectbox("z-axis", equalAttribute, index=2, key="z3") 
    fig3 = px.scatter_3d(df, x=x3, y=y3, z=z3, color="Bundle", hover_name="movieNm", 
                         category_orders={"Bundle": symbol[:k]}) 
    fig3.update_traces(marker=dict(size=2, opacity=0.7)) 
    fig3.update_layout(height=560, legend=dict(orientation="h")) 
    st.plotly_chart(fig3, width="stretch") 
    st.caption("Drag with mouse to rotate. Hover mouse over a dot to see the title.") 

st.subheader("Number of episodes per bundle and average of the four attributes") 
Summary = df.groupby("Bundle")[original unit].mean().round(2) 
Summary.insert(0, "Number of episodes", df.groupby("Bundle").size()) 
Summary = Summary.reindex(symbol[:k]) 
st.dataframe(Summary, width="stretch", column_config={ 
    "Number of episodes": st.column_config.NumberColumn("Number of episodes", format="%,d episodes"), 
    "Number of screens": st.column_config.NumberColumn("Number of screens", format="%.1f items"), 
    "Cumulative audience": st.column_config.NumberColumn("Cumulative audience", format="%,.0f people"), 
    "Number of days in top 10": st.column_config.NumberColumn("Number of days in top 10", format="%.1f days"), 
    "Long-run index": st.column_config.NumberColumn("Long-run index", format="%.2f times"), 
}) 
st.caption("The mean is the original unit before taking the logarithm. The long-run index is the cumulative audience divided by the first-week audience and truncated at 20.") 

st.subheader("Five films with the most audiences in each group") 
column = st.columns(k) 
for i, name in enumerate(symbol[:k]): 
    group = df[df["group"] == name].sort_values("cumulative audience", ascending=False).head(5) 
    column[i].markdown(f"**{name} group ({len(df[df['group'] == name]):,}films)**") 
    column[i].dataframe(pd.DataFrame({"movie": group["movieNm"].to_numpy(), 
                                "cumulative audience": group["cumulative audience"].to_numpy()}), 
                   width="stretch", hide_index=True, 
                   column_config={"cumulative audience": st.column_config.NumberColumn("cumulative audience", format="%,d명")})
