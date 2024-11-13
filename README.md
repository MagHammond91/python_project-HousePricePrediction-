## OUTLINE

- Creating a house-price prediction project

- Performing an initial exploratory data analysis with Jupyter notebook

- Setting up a train, validation and test framework

- Testing different models to find the best fit 

- Using the model to predict house prices



## PROJECT DESCRIPTION

The aim of this project is to develop a machine learning model that can accurately predict the price of a house based on various features such as area, bathrooms, bedrooms, stories and other relevant factors. The project will involve collecting a dataset of house listings with associated prices and features from online sources or existing databases. Data preprocessing steps will be implemented to prepare the dataset for modeling, including encoding categorical variables, and scaling numerical features.

Next, various machine learning algorithms such as random forest, Naive Base, K-Nearest Neighbor, Gradient Boosting Classifier,      logistic regression, support vector machine  will be trained and evaluated to identify the best-performing model. Feature importance analysis will also be conducted to understand which features have the most significant impact on the predicted house prices.

The developed model will be deployed into a user-friendly interface, allowing users to input house features and obtain a predicted price estimate. Additionally, the project will include documentation detailing the steps involved in data loading, preprocessing, modeling, evaluation, and deployment, making it accessible for others to understand and replicate.



## DATASET DESCRIPTION

This dataset provides comprehensive information for house price prediction, with 13 column names



**DATASET URL**
https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/data


## ATTRIBUTES

**Price**: The price of the house.

**Area**: The total area of the house in square feet.

**Bedrooms**: The number of bedrooms in the house.

**Bathrooms**: The number of bathrooms in the house.

**Stories**: The number of stories in the house.

**Mainroad**: Whether the house is connected to the main road (Yes/No).

**Guestroom**: Whether the house has a guest room (Yes/No).

**Basement**: Whether the house has a basement (Yes/No).

**Hot water heating**: Whether the house has a hot water heating system (Yes/No).

**Airconditioning**: Whether the house has an air conditioning system (Yes/No).

**Parking**: The number of parking spaces available within the house.

**Prefarea**: Whether the house is located in a preferred area (Yes/No).

**Furnishing status**: The furnishing status of the house (Fully Furnished, Semi-Furnished, Unfurnished).


IMPORT LIBRARIES
```
## for loading and preprocessing 
import pandas as pd
import numpy as np 

## for data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

## splitting dataset and feature engineering 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pickle
```

## THE ANALYSIS

### OBSERVATION 

- There are 545 rows and 13 columns in the dataset.

- The data type of the columns are both interger and object.

- The columns in the datasets are:
'price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea', 'furnishingstatus'

- There are no missing values and duplicates.

- Column names remnamed for easy reading purposes.


**Check codes out here : [project_HousePricePredictionModel](/project_HousePricePrediction.ipynb)**

### EXPLORATORY DATA ANALYSIS

- Statistical summary of the numeric coulums

| Statistic | Price       | Area   | Bedrooms | Bathrooms | Stories | Parking |
|-----------|-------------|--------|----------|-----------|---------|---------|
| Count     | 545.0       | 545.0  | 545.0    | 545.0     | 545.0   | 545.0   |
| Mean      | 4,766,729.0 | 5,151.0| 3.0      | 1.0       | 2.0     | 1.0     |
| Std       | 1,870,440.0 | 2,170.0| 1.0      | 1.0       | 1.0     | 1.0     |
| Min       | 1,750,000.0 | 1,650.0| 1.0      | 1.0       | 1.0     | 0.0     |
| 25%       | 3,430,000.0 | 3,600.0| 2.0      | 1.0       | 1.0     | 0.0     |
| 50%       | 4,340,000.0 | 4,600.0| 3.0      | 1.0       | 2.0     | 0.0     |
| 75%       | 5,740,000.0 | 6,360.0| 3.0      | 2.0       | 2.0     | 1.0     |
| Max       | 13,300,000.0| 16,200.0| 6.0     | 4.0       | 4.0     | 3.0     |

*Statistical Summary Table*

- Converting the non-merical columns to numerical values for analysis purposes.
```
df['main_road'] = df['main_road'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['guestroom'] = df['guestroom'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['basement'] = df['basement'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['hotwater_heating'] = df['hotwater_heating'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['air_conditioning'] = df['air_conditioning'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['pref_area'] = df['pref_area'].\
map( {'no': 0, 'yes': 1 } ).astype(int)

df['furnishing_status'] = df['furnishing_status'].\
map( {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0 } ).astype(int)
```
- Performing a correlation on all columns to identify the relationship between the columns.The correlation matrix will be visualised using a heatmap.

```
corr_matrix = df.corr()
corr_matrix

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for All Columns')
plt.show()
```
![Correlation Matrix](/)
*Heatmap visualizing the correlation matrix of all columns*