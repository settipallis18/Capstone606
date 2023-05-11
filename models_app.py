# How to run
# pip install streamlit
# streamlit run models_app.py


import streamlit as st
import warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy
import matplotlib.colors as mcolors
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Hide warnings
warnings.filterwarnings("ignore")

# Read Data
df = pd.read_csv("Global_Earthquake_Data.csv")


@st.cache
def impute_missing_values(df):
    # Remove rows with missing values in the 'depth' column
    df = df.dropna(subset=['depth'])

    # Replace missing values in 'magType' with the most common value
    most_common_magType = df['magType'].mode()[0]
    df['magType'].fillna(most_common_magType, inplace=True)

    # Replace missing values in 'nst' with 0
    df['nst'].fillna(0, inplace=True)

    # Replace missing values in 'gap' with the median value
    median_gap = df['gap'].median()
    df['gap'].fillna(median_gap, inplace=True)

    # Replace missing values in 'dmin' with the mean value
    mean_dmin = df['dmin'].mean()
    df['dmin'].fillna(mean_dmin, inplace=True)

    # Replace missing values in 'rms' with the median value
    median_rms = df['rms'].median()
    df['rms'].fillna(median_rms, inplace=True)

    # Replace missing values in 'horizontalError' with the median value
    median_horizontalError = df['horizontalError'].median()
    df['horizontalError'].fillna(median_horizontalError, inplace=True)

    # Replace missing values in 'depthError' with the median value
    median_depthError = df['depthError'].median()
    df['depthError'].fillna(median_depthError, inplace=True)

    # Replace missing values in 'magError' with the median value
    median_magError = df['magError'].median()
    df['magError'].fillna(median_magError, inplace=True)

    # Replace missing values in 'magNst' with the median value
    median_magNst = df['magNst'].median()
    df['magNst'].fillna(median_magNst, inplace=True)

    # Replace missing values in 'place' with 'Unknown'
    df['place'].fillna('Unknown', inplace=True)

    return df


# Preprocess data
df = impute_missing_values(df)


# EDA and Visualizations
def eda_visualizations():
    # Filter columns for analysis
    df2 = df[['time', 'latitude', 'longitude', 'mag']]

    # Convert object to datetime
    df2['time'] = pd.to_datetime(df2['time'])

    # Display datatypes
    st.subheader('Data Types')
    st.dataframe(df2.dtypes)

    # Display filtered data
    st.subheader('Filtered Data')
    st.dataframe(df2.head())

    # Plotting
    st.subheader('Magnitude Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(df2['mag'], kde=True)
    st.pyplot()


# Segregating Columns for Prediction
df3 = df[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst', 'mag']]

# Linear Regression Model
def linear_regression_model():
    # Feature selection
    X = df3[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
    y = df3['mag']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Plot evaluation
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Linear Regression Model Evaluation')
    st.pyplot()

    # Display mean squared error
    st.subheader('Linear Regression Model')
    st.write(f"Mean Squared Error: {mse}")


# OLS Model
def ols_model():
    # Add a constant term to the input features
    X = sm.add_constant(df3[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']])
    y = df3['mag']

    # Define and fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Make predictions
    y_pred = results.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)

    # Plot evaluation
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('OLS Model Evaluation')
    st.pyplot()

    # Display mean squared error
    st.subheader('OLS Model')
    st.write(f"Mean Squared Error: {mse}")


# Other Regression Models
# Other Regression Models
def other_regression_models():
    # Define the regression models
    models = [
        Ridge(),
        Lasso(),
        XGBRegressor()
    ]

    for model in models:
        # Model training
        model.fit(df3[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']], df3['mag'])
        y_pred = model.predict(df3[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']])
        mse = mean_squared_error(df3['mag'], y_pred)

        # Plot evaluation
        plt.scatter(df3['mag'], y_pred, alpha=0.5)
        plt.plot([min(df3['mag']), max(df3['mag'])], [min(df3['mag']), max(df3['mag'])], 'r--', lw=2)
        plt.xlabel('Actual Magnitude')
        plt.ylabel('Predicted Magnitude')
        plt.title(f'{model.__class__.__name__} Model Evaluation')
        st.pyplot()

        # Display mean squared error
        st.subheader(f'{model.__class__.__name__} Model')
        st.write(f"Mean Squared Error: {mse}")


# Streamlit App
def main():
    st.title('Earthquake Data Analysis')
    # Display data info
    st.subheader('Data Info')
    st.dataframe(df.info())

    # Check NA values
    st.subheader('NA Values')
    st.dataframe(df.isna().sum())

    # EDA and Visualizations
    eda_visualizations()

    # Linear Regression Model
    linear_regression_model()

    # OLS Model
    ols_model()

    # Other Regression Models
    other_regression_models()


if __name__ == '__main__':
    main()
