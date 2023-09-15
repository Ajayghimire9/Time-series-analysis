#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from deep_translator import GoogleTranslator


# In[4]:





# In[8]:


df = pd.read_excel('h22index-42.xlsx', na_values=[r'-', r'–', r'－'], header=1, skipfooter=6, parse_dates = True)
translator = GoogleTranslator(src='japanese', dest='en')
df.rename(columns=translator.translate, inplace=True)
df['Unnamed: 0'] = df['Unnamed: 0'].apply(translator.translate)

# It seems that first column contains dates. So we can change the name of the column as Date.
# Also we can make some changes on column names for easy use

df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
df.columns = df.columns.str.title()
df.columns = df.columns.str.replace(' ','_')
# Extracting date string from the translated first column values

col = [date[8:] for date in df['Date']]
col[0:3]

# Converting strings to Date Format

df['Date'] = pd.to_datetime(col, format='%B %d, %Y')
df.head()

# Setting "Date" column as the index to have an ordered time-series.

df.set_index(keys='Date', inplace=True)
df.head()


# Pre processing
df.isna().sum()

''' In total, we have 249 records in our dataset.
When we observe missing values, we can see that Eggplant and Spinach have the most missing values with 200 and 199 respectively.
Carrots has 146 missing values which is approximately 60% of the records.
Chinese_Cabbage, Radish, Cucumber, and Potatoes have closer in terms of their number of missing values.'''
# checking Missing indexes 
# check for missing dates in the index    

missing_weeks = pd.date_range(start='2017-11-06', end='2022-10-10', freq='W-MON').difference(df.index)
print(f'The missing weeks in the dataset are:\n{missing_weeks}')


'''As we can observe there are some missing weeks in our dataset. 
They might have happened for many reasons like the Covid pandemic, and the lack of information.'''


# check if any row has missing value for all columns
df[df.isnull().all(axis=1)]

'''Also, there are 2 weeks that don't have any values for any variables.'''
#  Adding missing weeks into dataframe
# build a temporary dataframe with missing weeks as index
# and the same columns as the main dataset
tmp_df = pd.DataFrame(index=missing_weeks, columns=df.columns)

# concat this temporary dataframe to the main dataset
df = pd.concat([df, tmp_df])

# sort the dataframe according to the index
df.sort_index(inplace=True)

# set index to weekly period with 'Monday' as the starting day of the week
df.index.to_period('W-MON')

# visulaizing missing values 
# Plotting Variables through time

for column in df.columns:

    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df.index.min()
    end_date = df.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df[column], color='blue')

    # adds title to our time series plot
    plt.title(column)
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()
    
# visualise the pattern of missing values
msno.matrix(df, figsize=(25,12), color=(0.8, 0.3, 0.3))

# visualise the correlation between the variables in terms of occurance of missing value
msno.heatmap(df, figsize=(25,12))
'''The broken points within the curve indicate missing values in our data. We can observe some seasonilities in some variables like Cucumber,Chinese Cabbage, and Raddish.
Chinese cabbage, and Radish have missing data during opposite intervals compared to Cucumber.
green onion and onion had missing data during the initial times.
There are several methods for imputing missing values in time series data. Which method is most appropriate will depend on the characteristics of your data and the reason for the missing values. Here are a few common techniques:

Interpolation: This method involves estimating the missing values using the known values surrounding the missing value. Linear interpolation is a simple form of this method, where the missing value is estimated to be the weighted average of the values on either side. More sophisticated interpolation methods, such as spline interpolation, can be used to better capture the underlying trend in the data.

Last Observation Carried Forward (LOCF): This method involves replacing the missing value with the last known value. This can be a simple and effective approach if the missing values are not too numerous and the data are not changing too rapidly.

Next Observation Carried Backward (NOCB): It is a method used in time series analysis for estimating the values of a variable at a given point in time using the most recent observation and the observations that come after it. This method is typically used when the data are not evenly spaced in time, or when there are missing values in the data.

Mean/Median/Mode Imputation: This method involves replacing the missing value with the mean, median, or mode of the known values. This can be a simple and effective approach if the missing values are not too numerous and the data are not too noisy.

It's important to carefully evaluate the performance of each method and choose the one that best fits the characteristics of your data. In some cases, it may be necessary to try multiple methods and compare the results.'''

#  Mean Imputation
# For Mean Imputation, I create a copy of original dataframe
df1 = df.copy()
for column in df1.columns:
    print('The mean value of {} = {}.'.format(column,df1[column].mean()))
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df1.index.min()
    end_date = df1.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df1[column], color='blue')

    # adds title to our time series plot
    plt.title(column + ' - Before Mean Imputation')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()
    # Calculate the mean value of each selected column
    impute_values = df1[column].mean()
    # Replace missing values with the calculated mean value
    df1[column] = df1[column].fillna(impute_values)
    # pass the data and declared the colour of your curve, i.e., red
    plt.plot(df1[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After Mean Imputation')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.axhline(y=df1[column].mean(),ls=('--'))
    # print the plot
    plt.show()
    
    
# 2 median imputation 
# For Median Imputation, I create a copy of original dataframe
df2 = df.copy()

for column in df2.columns:
    print('The median value of {} = {}.'.format(column,df2[column].median()))
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df2.index.min()
    end_date = df2.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df2[column], color='blue')

    # adds title to our time series plot
    plt.title(column + ' - Before Median Imputation')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()
    # Calculate the median value of each selected column
    impute_values = df2[column].median()
    # Replace missing values with the calculated median value
    df2[column] = df2[column].fillna(impute_values)
    # pass the data and declared the colour of your curve, i.e., red
    plt.plot(df2[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After Median Imputation')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.axhline(y=df2[column].median(),ls=('--'))
    # print the plot
    plt.show()
    
# 3. Last Observation Carried Forward (LOCF)
#According to this technique, the missing value is imputed using the values before it in the time series.
# For LOCF, I create a copy of original dataframe
df3 = df.copy()
for column in df3.columns:
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)
    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df3.index.min()
    end_date = df3.index.max()
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # plots our series
    plt.plot(df3[column], color='blue')
    # adds title to our time series plot
    plt.title(column + ' - Before LOCF')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()

# LOCF
    df3[column] = df3[column].fillna(method ='ffill')
# pass the data and declared the colour of your curve, i.e., red
    plt.plot(df3[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After LOCF')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    # print the plot
    plt.show()
    
# 4. Next Observation Carried Backward (NOCB)
#According to this technique, the missing values are imputed using an immediate value ahead of them.

# For NOCB, I create a copy of original dataframe
df4 = df.copy()

for column in df4.columns:
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df4.index.min()
    end_date = df4.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df4[column], color='blue')

    # adds title to our time series plot
    plt.title(column + ' - Before NOCB')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()
# NOCB
    df4[column] = df4[column].fillna(method ='bfill')
# pass the data and declared the colour of your curve, i.e., red
    plt.plot(df4[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After NOCB')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    # print the plot
    plt.show()
    
# 5. Linear Interpolation
#This technique originates from Numerical Analysis, which estimates unknown values by assuming linear relation within a range of data points, unlike linear extrapolation, which estimates data outside the range of the provided data points. To estimate the missing values using linear interpolation, we look at the past and the future data from the missing value.
# For Linear Interpolation, I create a copy of original dataframe
df5 = df.copy()

for column in df5.columns:
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df5.index.min()
    end_date = df5.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df5[column], color='blue')

    # adds title to our time series plot
    plt.title(column + ' - Before Linear Interpolation')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # print the plot
    plt.show()
# Linear Interpolation
    df5[column] = df5[column].interpolate(method='linear')
# pass the data and declared the colour of your curve, i.e., red
    plt.plot(df5[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After Linear Interpolation')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    # print the plot
    plt.show()
    
'''6. Polynomial Interpolation
Polynomial interpolation is a method of estimating the value of a variable by fitting a polynomial curve to a set of data points.
It can be used to interpolate new values along the curve.'''

# For Polynomial interpolation, I create a copy of original dataframe
df6 = df.copy()

for column in df5.columns:
    # set the size of our plot
    plt.rcParams['figure.figsize']=(9,3)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df6.index.min()
    end_date = df6.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # plots our series
    plt.plot(df6[column], color='blue')

    # adds title to our time series plot
    plt.title(column + ' - Before Polynomial Interpolation')
    plt.xlabel('Time')
    plt.ylabel('Value')
# print the plot
    plt.show()
# Polynomial interpolation
    df6[column] = df6[column].interpolate(method='cubic')
# pass the data and declared the colour of your curve, i.e., red
    plt.plot(df6[column], color='red')
    # add tittle to the plot
    plt.title(column + ' - After Polynomial Interpolation')
    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)
    # adds title to our time series plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    # print the plot
    plt.show()
    
    
''' 1. Cabbage: LOCF, NOCB, LI, PI 2.Green_Onion: Mean, Median, PI  3.Lettuce: LOCF, NOCB, LI, PI  4.Potatoes: NOCB,  5.Onion: LOCF, NOCB, LI,
6.Cucumber: Mean, Median, LOCF,
7.Tomato: LOCF, NOCB, LI, PI
8.Spinach: Mean,Median, PI
9.Carrot: Mean,Median, LOCF, NOCB, LI,
10.Chinese_Cabbage: LOCF, NOCB, LI
11.Radish: NOCB,
12.Eggplant: Mean,Median, PI
- Applying Chosen Methods to Fill Out Missing Values
1.Cabbage: LOCF*
2.Green_Onion: PI* (+ median imputation for initials)
3.Lettuce: NOCB*
4.Potatoes: LI* (+ NOCB for initials)
5.Onion: NOCB*
6.Cucumber: LI* (+ mean imputation for initials)
7.Tomato: PI*
8.Spinach: ???
9.Carrot: ???
10.Chinese_Cabbage: LI*
11.Radish: LI*
12.Eggplant: ???'''


# In[11]:


import matplotlib.pyplot as plt

# Use a different style for the plots
plt.style.use('seaborn-darkgrid')

# Create a color palette
palette = plt.get_cmap('tab10')

# Drop 'Spinach', 'Eggplant', and 'Carrot' if they exist in the dataframe
columns_to_drop = ['Spinach', 'Eggplant', 'Carrot']
df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

# Determine the number of columns to plot
num_columns = len(df.columns)

# Set the size of our plot
plt.figure(figsize=(15, 3 * num_columns))

for idx, column in enumerate(df.columns, start=1):
    plt.subplot(num_columns, 1, idx)

    # Determine the start and end dates of the time period that includes the missing values in a time series plot
    start_date = df.index.min()
    end_date = df.index.max()

    # Set the range of the x-axis to the time period
    plt.xlim(start_date, end_date)

    # Plot the original data using the color palette
    plt.plot(df[column], color=palette(idx), linewidth=2.5, label='Before Imputation')

    # Imputation methods remain unchanged...
    
    # Plot the imputed data
    plt.plot(df[column], color='Green', linestyle='--', linewidth=2.5, label='After Imputation')
    
    # Set title, labels, and legend
    plt.title(column, fontsize=14, fontweight='bold', color=palette(idx))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show all the plots
plt.show()


# In[ ]:


#Checking Missing Values After Imputation:
df.isna().sum()
# Round the values in the dataframe to 1 decimal place
df = df.round(1)
df.to_csv('dataframe.csv')


# In[12]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Extract the dates as a separate feature
dates = df.index

# Create an empty DataFrame to store the future predictions
future_predictions_df = pd.DataFrame(index=pd.date_range(start=dates[-1] + pd.DateOffset(days=1), periods=7))

# Iterate over each column in the DataFrame
for column in df.columns:
    # Extract the target variable and create a numerical representation of dates
    target = df[column]
    numeric_dates = (dates - dates.min()).days.values.reshape(-1, 1)  # Use .days to get the numerical representation

    # Create a linear regression model
    linear_model = LinearRegression()

    # Fit the linear regression model
    linear_model.fit(numeric_dates, target)

    # Generate predictions for future dates
    future_numeric_dates = np.arange(len(dates), len(dates) + len(future_predictions_df)).reshape(-1, 1)
    future_predictions = linear_model.predict(future_numeric_dates)

    # Add the future predictions to the DataFrame
    future_predictions_df[column] = future_predictions

# Print the future predictions
print(future_predictions_df)
# Concatenate the input DataFrame with the future predictions DataFrame
combined_df = pd.concat([df, future_predictions_df])
# Print the combined DataFrame
print(combined_df)

# Plot the graph
plt.figure(figsize=(10, 6))  # Set the figure size
for column in combined_df.columns:
    plt.plot(combined_df.index, combined_df[column], label=column)

plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Value')  # Set the y-axis label
plt.title('Future Predictions')  # Set the title
plt.legend()  # Show the legend
plt.grid(True)  # Add gridlines
plt.show()  # Show the plot


# In[13]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Extract the dates as a separate feature
dates = df.index

# Create an empty DataFrame to store the future predictions
future_predictions_df = pd.DataFrame(index=pd.date_range(start=dates[-1] + pd.DateOffset(days=1), periods=7))

# Iterate over each column in the DataFrame
for column in df.columns:
    # Extract the target variable and create a numerical representation of dates
    target = df[column]
    numeric_dates = (dates - dates.min()).days.values.reshape(-1, 1)  # Use .days to get the numerical representation

    # Create a linear regression model
    linear_model = LinearRegression()

    # Fit the linear regression model
    linear_model.fit(numeric_dates, target)

    # Generate predictions for future dates
    future_numeric_dates = np.arange(len(dates), len(dates) + len(future_predictions_df)).reshape(-1, 1)
    future_predictions = linear_model.predict(future_numeric_dates)

    # Add the future predictions to the DataFrame
    future_predictions_df[column] = future_predictions

# Print the future predictions
print(future_predictions_df)

# Combine the original DataFrame and the future predictions DataFrame
combined_df = pd.concat([df, future_predictions_df])

# Plot the graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Set the figure size
for column in combined_df.columns:
    plt.plot(combined_df.index, combined_df[column], label=column)

plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Value')  # Set the y-axis label
plt.title('Future Predictions')  # Set the title
plt.legend()  # Show the legend
plt.grid(True)  # Add gridlines
plt.show()  # Show the plot


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Extract the dates as a separate feature
dates = df.index

# Create an empty DataFrame to store the future predictions
future_predictions_df = pd.DataFrame(index=pd.date_range(start=dates[-1] + pd.DateOffset(days=1), periods=7))

# Iterate over each column in the DataFrame
for column in df.columns:
    # Extract the target variable and create a numerical representation of dates
    target = df[column]
    numeric_dates = (dates - dates.min()).days.values.reshape(-1, 1)  # Use .days to get the numerical representation

    # Create a random forest regression model
    rf_model = RandomForestRegressor()

    # Fit the random forest regression model
    rf_model.fit(numeric_dates, target)

    # Generate predictions for future dates
    future_numeric_dates = np.arange(len(dates), len(dates) + len(future_predictions_df)).reshape(-1, 1)
    future_predictions = rf_model.predict(future_numeric_dates)

    # Add the future predictions to the DataFrame
    future_predictions_df[column] = future_predictions

# Print the future predictions
print(future_predictions_df)

# Combine the original DataFrame and the future predictions DataFrame
combined_df = pd.concat([df, future_predictions_df])

# Plot the graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Set the figure size
for column in combined_df.columns:
    plt.plot(combined_df.index, combined_df[column], label=column)

plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Value')  # Set the y-axis label
plt.title('Future Predictions')  # Set the title
plt.legend()  # Show the legend
plt.grid(True)  # Add gridlines
plt.show()  # Show the plot

# Create a SHAP explainer object
explainer = shap.Explainer(rf_model)
# Generate SHAP values for the entire dataset
shap_values = explainer.shap_values(numeric_dates)
# Plot the SHAP summary plot
shap.summary_plot(shap_values, numeric_dates, feature_names=['Date'])
# Create a LIME explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(numeric_dates, mode="regression")
# Explain an individual prediction
explanation = explainer.explain_instance(numeric_dates[0], rf_model.predict, num_features=numeric_dates.shape[1])
explanation.show_in_notebook()

