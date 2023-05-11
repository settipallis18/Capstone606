# How to run
# pip install streamlit
# streamlit run randomForest_app.py


# Import the required packages:
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Define a function to read the data from the CSV file:
def read_data():
    df = pd.read_csv("/Users/settipallis154/Desktop/CAPSTONE/Global_Earthquake_Data.csv")
    return df

# Define a function to perform data preprocessing:
def preprocess_data(df):
    df = df.dropna(subset=['mag', 'depth'])
    most_common_magType = df['magType'].mode()[0]
    df['magType'].fillna(most_common_magType, inplace=True)
    df['nst'].fillna(0, inplace=True)
    median_gap = df['gap'].median()
    df['gap'].fillna(median_gap, inplace=True)
    mean_dmin = df['dmin'].mean()
    df['dmin'].fillna(mean_dmin, inplace=True)
    median_rms = df['rms'].median()
    df['rms'].fillna(median_rms, inplace=True)
    median_horizontalError = df['horizontalError'].median()
    df['horizontalError'].fillna(median_horizontalError, inplace=True)
    median_depthError = df['depthError'].median()
    df['depthError'].fillna(median_depthError, inplace=True)
    median_magError = df['magError'].median()
    df['magError'].fillna(median_magError, inplace=True)
    median_magNst = df['magNst'].median()
    df['magNst'].fillna(median_magNst, inplace=True)
    df['place'].fillna('Unknown', inplace=True)
    return df


# Define the main function for your Streamlit app and the code to be executed:
def main():
    # Set the page title
    st.title("Random Forest Regression")

    # Load your dataset or define your input features (X) and target variable (y)
    df = read_data()
    df = preprocess_data(df)

    # Feature selection
    X = df[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError',
            'magNst']]

    # Input features
    y = df['mag']  # Target variable

    # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define the parameter grid for grid search
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 4, 8],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }

    # Create an instance of the RandomForestRegressor model
    rf_model = RandomForestRegressor()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    st.subheader("Best Hyperparameters:")
    st.write(best_params)

    st.subheader("Best Model:")
    st.write(best_model)

    # Make predictions using the best model
    y_pred = best_model.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)
    st.subheader("Mean Squared Error:")
    st.write(mse)

    # Plot the evaluation
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
    ax.set_xlabel('Actual Magnitude')
    ax.set_ylabel('Predicted Magnitude')
    ax.set_title('Best Model Evaluation')
    st.pyplot(fig)


if __name__ == '__main__':
    main()
