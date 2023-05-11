# app.py
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

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

tab1, tab2, tab3 , tab4 = st.tabs(["Analysis","Prediction","Randomforest", "Models"])
# Reading Data
df = pd.read_csv("/Users/settipallis154/Desktop/CAPSTONE/Global_Earthquake_Data.csv")

# Check NA values
df = df.dropna(subset=['mag', 'depth'])
df['time'] = pd.to_datetime(df['time'])
df2 = df[['time', 'latitude', 'longitude', 'mag']]

with tab1:
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
        # plt.figure(figsize=(20, 20))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        # ax.coastlines()
        # p = ax.scatter(x=df2['longitude'], y=df2['latitude'], c=df2['mag'].sort_values(), cmap='Reds', alpha=0.5, s=15)
        # st.pyplot()

with tab2:
        # ### Imputation
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


        # Replace missing values in 'rms' with the median value
        median_horizontalError = df['horizontalError'].median()
        df['horizontalError'].fillna(median_horizontalError, inplace=True)

        # Replace missing values in 'rms' with the median value
        median_depthError = df['depthError'].median()
        df['depthError'].fillna(median_depthError, inplace=True)

        # Replace missing values in 'rms' with the median value
        median_magError = df['magError'].median()
        df['magError'].fillna(median_magError, inplace=True)

        # Replace missing values in 'rms' with the median value
        median_magNst = df['magNst'].median()
        df['magNst'].fillna(median_magNst, inplace=True)



        # Replace missing values in 'place' with 'Unknown'
        df['place'].fillna('Unknown', inplace=True)


        # ### We have not Imputer the Error Features as they are More than 30% and Imputing those will Skew or diviate our data.
        df.isna().sum()


        # # EDA and Visualizations

        # ### For analysis lets filter some columns
        df2 = df[['time', 'latitude', 'longitude', 'mag']]
        df2.head()


        # #### Viewing datatypes
        df2.dtypes


        # #### Converting object to datetime
        df2['time'] = pd.to_datetime(df2['time'])


        # #### Data Description
        df2.describe()
        # ### The average earthquake magnitude for the time period was 4.94, and the average depth was 69.79 kilometres. Notably, the strongest earthquake with a recorded magnitude during this time was 9.5. Let's look more closely at this important event's qualities to better comprehend it and its significance.


        max_index = df2['mag'].idxmax() # Creating a variable of the max magnitude 
        max_data = df2.loc[max_index, :] # storing the row of the largest earthquake
        max_data
        # ### The strongest earthquake that occurred during the time period under consideration had a magnitude of 9.5 and occurred on May 22, 1960, at a depth of 25 kilometres. Let's create a graphic that shows the earthquake's epicentre on the Earth's surface to obtain further understanding of its position and size.
        # 

        # ## Bar plot of Disaster types
        # We can see that Earthquake is most common form of Natural Disaster.

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='type')
        plt.xlabel('Disaster Type')
        plt.ylabel('Count')
        plt.title('Distribution of Disaster Types')
        plt.xticks(rotation=45)
        plt.show()


        # ### Line plot of earthquake frequency over time
        # We can see that the Frequency of Earthquakes increases over time.
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year
        earthquake_counts = df['year'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.plot(earthquake_counts.index, earthquake_counts.values)
        plt.xlabel('Year')
        plt.ylabel('Earthquake Count')
        plt.title('Earthquake Frequency Over Time')
        plt.xticks(rotation=45)
        plt.show()


        # ### Bar plot of earthquake status
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='status')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.title('Distribution of Earthquake Status')
        plt.xticks(rotation=45)
        plt.show()


        #
        plt.figure(figsize=(15, 15)) # creating a figure with size being 20 by 20

        ax = plt.axes(projection=ccrs.PlateCarree()) # Projecting PlateCarree from the cartopy library

        ax.set_extent((-180,180,-90,90), crs = ccrs.PlateCarree()) # Showing entire globe in PlateCarree format

        gl = ax.gridlines(linestyle='-.', draw_labels=True) # adding gridlines to the plot
        gl.top_labels = False # removing the top label
        gl.right_labels = False # removing the right label

        ax.coastlines() # adding coastlines to the plot

        p = ax.scatter(x=max_data['longitude'], y=max_data['latitude'], s=250, c='r') # Plotting the max magnitude data point

        plt.title('The Largest Earthquake in the Past 116 Years was on May 22, 1960', fontsize=15)

        plt.show()


        # ### Examining All Earthquakes
        plt.figure(figsize=(20, 20))
        ax = plt.axes(projection=ccrs.PlateCarree())

        gl = ax.gridlines(linestyle='-.', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.coastlines()

        p = ax.scatter(x=df2['longitude'], y=df2['latitude'], c =df2['mag'].sort_values(), cmap='Reds', alpha=0.5, s=15)
        plt.title('Earthquake Magnitude by Location From September 1906-2023', fontsize=18)


        plt.ylabel('Magnitude', fontsize=16)

        plt.show()
        # ### The locations of all earthquakes are shown on the figure above, with varying shades of red denoting each earthquake's magnitude. Stronger earthquakes are represented by redder hues. The figure clearly shows where the plate borders are, which is where most earthquakes tend to happen.
        # 


        # ### Let's look at the locations of all earthquakes with a magnitude of 8 or above to have a better understanding of the most severe earthquakes in the last 116 years.
        strong_earthequakes = df2[df2['mag']>=8] # create an new Dataframe with earthquakes of 8 or greater
        strong_earthequakes.shape # looking at the amount of elements in the new DF

        # ### These results illustrate a total of 97 significant earthquakes that meet our criteria of a magnitude of 8 or higher.



        # Lets take a quick look at a histogram of the strong data magnitude
        plt.hist(strong_earthequakes['mag'])

        plt.show()


        # ### This discovery is particularly noteworthy because there are relatively few earthquakes measuring 8.6 or higher, and it appears to follow an exponential distribution. Let's create a plot that shows the locations of these occurrences so we may examine this pattern in more detail.
        # 


        plt.figure(figsize=(15, 15))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent((-180,180,-90,90), crs = ccrs.PlateCarree()) # Zooming in on the graph

        gl = ax.gridlines(linestyle='-.', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.coastlines()

        p = ax.scatter(x=strong_earthequakes['longitude'], y=strong_earthequakes['latitude'], c =strong_earthequakes['mag'], cmap='Reds', s=140)
        plt.title('Earthquakes with a Magnitude of 8 or Greater from 1906-2023', fontsize=18)


        plt.show()



        # ### According to the data, seismic activity is most intense close to tectonic plate borders. It's interesting to note that the dataset's strongest earthquakes were found to occur around Indonesia, Japan, and the western coast of South America. Notably, it is a significant finding that the majority of these strongest earthquakes occurred at convergent plate boundaries.



        # lets plot
        plt.figure(figsize=(14, 8))
        plt.scatter(strong_earthequakes['time'], strong_earthequakes['mag'])
        plt.title("Magnitude over Time")
        plt.xlabel("Year")
        plt.ylabel("Magnitude")
        plt.show()




        df_size = len(df['time'])
        df_size

        strong_earthequakes_size = len(strong_earthequakes['time'])
        strong_earthequakes_size

        print('{} is the percentage of strong earthquakes to all earthquakes >4.5 magnitude'.format((strong_earthequakes_size / df_size) * 100))



        # ## Analysis Outcome:
        # 
        # ### The fact that severe earthquakes aren't always related to a certain amount of time is one of the analysis's most intriguing findings. Strong earthquakes, which are very uncommon occurrences with a likelihood of occuring just 0.034 percent of the time in earthquakes that measure 4.5 or above, are difficult to identify any distinct patterns linked with. A noteworthy fact is that the bulk of the largest earthquakes are found close to convergent plate borders, suggesting that these areas are more vulnerable to seismic activity. 
        # 
        # 
        # ### This dataset may be further analysed in a number of different ways. For instance, one may isolate the data for a particular area and determine the likelihood that a severe earthquake would occur there, offering important insights for risk management and disaster planning. Additionally, an interactive dashboard created with Tableau or a similar programme could show how uncommon these events are as well as how strong earthquakes tend to cluster near convergent plate boundaries. Our knowledge of earthquake behaviour might be substantially improved by combining this dataset with another one that has details about the type of plate boundary at a specific site.
        # 



        # ## Segregating Columns for Prediction
        df3 = df[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst', 'mag']]
        df3.head()


        # ### First Lets Train with the Linear Reg Model to see how its fit
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error



        # Feature selection
        X = df[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
        # Input features
        y = df['mag']  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Machine Learning Model (Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('Mean Squared Error:', mse)



        # Plotting the evaluation
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        plt.xlabel('Actual Magnitude')
        plt.ylabel('Predicted Magnitude')
        plt.title('Linear Regression Model Evaluation')
        plt.show()


        # ## OLS Model
        import statsmodels.api as sm
        import statsmodels.formula.api as smf


        # Add a constant term to the input features
        X = sm.add_constant(X)

        # Define and fit the OLS model
        model = sm.OLS(y, X)
        results = model.fit()

        # Make predictions
        y_pred = results.predict(X)

        # Calculate the mean squared error
        mse = mean_squared_error(y, y_pred)
        print(f"OLS Model: Mean Squared Error - {mse}")

        # Plot the evaluation
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
        plt.xlabel('Actual Magnitude')
        plt.ylabel('Predicted Magnitude')
        plt.title('OLS Model Evaluation')
        plt.show()




        # # Training with more models
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error



        # %%time
        # Define the regression models
        models = [
            LinearRegression(),
            Ridge(),
            Lasso(),
            RandomForestRegressor(),
            XGBRegressor()
        ]

        # Train and evaluate the models
        for model in models:
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            print(f"{model.__class__.__name__}: Mean Squared Error - {mse}")

            # Plot the evaluation
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
            plt.xlabel('Actual Magnitude')
            plt.ylabel('Predicted Magnitude')
            plt.title(f'{model.__class__.__name__} Model Evaluation')
            plt.show()


        # ## From the Above Scores and charts we can see that RandomForrest converges and Fitted the Data Best.

        # ## Above we got the Random Forest Model with the best Parameters.


        # %%time
        from sklearn.model_selection import GridSearchCV

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100],
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto']
        }

        # # Define the parameter grid for hyperparameter tuning
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [5, 10, 15],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['auto', 'sqrt']
        # }

        # Create the RandomForestRegressor model
        rf_model = RandomForestRegressor()

        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
        grid_search.fit(X, y)

        # Print the best hyperparameters
        print("Best Hyperparameters:", grid_search.best_params_)


        # #### Testing the Model
        # dict(X.loc[0])


        data = {'const': [1.0],
        'latitude': [41.805],
        'longitude': [79.8675],
        'depth': [10.0],
        'nst': [46.0],
        'gap': [91.0],
        'dmin': [1.293],
        'rms':[0.8],
        'horizontalError': [6.59],
        'depthError': [1.897],
        'magError': [0.078],
        'magNst': [52.0]}



        # Get the best estimator from the grid search
        best_model = grid_search.best_estimator_

        # Define the custom data
        custom_data = pd.DataFrame(data)

        # Make predictions on the custom data using the best model
        predictions = best_model.predict(custom_data)

        # Print the predicted magnitude
        print('Predicted Magnitude:', predictions[0])

        print("Actual Value:",y.loc[0])



with tab3:
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
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto']
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




with tab4:
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
