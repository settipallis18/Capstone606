 # How to run
# pip install streamlit
# streamlit run visualize_app.py

# app.py
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy
import matplotlib.colors as mcolors
import seaborn as sb

# Reading Data
df = pd.read_csv("Global_Earthquake_Data.csv")

# Check NA values
df = df.dropna(subset=['mag', 'depth'])
df['time'] = pd.to_datetime(df['time'])
df2 = df[['time', 'latitude', 'longitude', 'mag']]

# Sidebar
st.sidebar.title("Earthquake Analysis")
analysis_type = st.sidebar.selectbox("Select Analysis", ("Data Summary", "Bar Plot", "Line Plot", "Map Plot"))

if analysis_type == "Data Summary":
    st.header("Data Summary")
    st.dataframe(df2.head())

    st.subheader("Data Description")
    st.write(df2.describe())

elif analysis_type == "Bar Plot":
    st.header("Bar Plot of Disaster Types")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='type')
    plt.xlabel('Disaster Type')
    plt.ylabel('Count')
    plt.title('Distribution of Disaster Types')
    plt.xticks(rotation=45)
    st.pyplot()

elif analysis_type == "Line Plot":
    st.header("Line Plot of Earthquake Frequency Over Time")
    df['year'] = df['time'].dt.year
    earthquake_counts = df['year'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(earthquake_counts.index, earthquake_counts.values)
    plt.xlabel('Year')
    plt.ylabel('Earthquake Count')
    plt.title('Earthquake Frequency Over Time')
    plt.xticks(rotation=45)
    st.pyplot()

elif analysis_type == "Map Plot":
    st.header("Map Plot of Earthquakes")
    plt.figure(figsize=(20, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
    ax.coastlines()
    p = ax.scatter(x=df2['longitude'], y=df2['latitude'], c=df2['mag'].sort_values(), cmap='Reds', alpha=0.5, s=15)
    st.pyplot()

