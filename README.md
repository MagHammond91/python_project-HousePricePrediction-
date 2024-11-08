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