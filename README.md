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
import joblib
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
- Performing a correlation on all columns to identify the relationship between the columns.

The correlation matrix will be visualised using a heatmap.

```
corr_matrix = df.corr()
corr_matrix

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for All Columns')
plt.show()
```
![Correlation Matrix](/images/output1.png)
*Heatmap visualizing the correlation matrix of all columns*


Given Price as our target variable and the rest of the columns as feature variables, it is realized that there is a fairly strong positive relationship between Price, and total area of the house and number of bathromms in the house, since their correlation co-effecient are over 0.50. 

Thus an increase in the total area and the number of bathrooms will result in an increase in price of the house.

### MACHINE LEARNING MODEL 

- Find the distribution and skewness of our target variable(price)
```
#Finding the distribution of our target variable(price)
sns.histplot(df['price'])

plt.title('Histogram Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.show()

#finding the skewness of the target variable(price)
df['price'].skew()
```
![Histogram Distribution of Price](/images/output2.png)
*Histogram visualizing the distribution of Price*

The distribution of price is positively shewed at 1.2122388370279802 and it needs to be normalized.

- Normalising the target variable(price) with the log function
```
#Normalising the target variable(price) with the log function
log_price = np.log1p(df['price'])

#Finding the distribution of our target variable(price)
sns.histplot(log_price)

plt.title('Histogram Distribution of the Log Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.show()

#finding the skewness of the target variable(price)
df1 = log_price.skew()
df1
```
![Histogram Distribution of Log Price](/images/output3.png)
*Histogram visualizing the distribution of normalized Price*

The distribution of price is now normally disctributed and skewed at 0.14086281102108905

- Setting up a train, validation and test framework.

The dataset of 545 rows will be split up into train, validation and test set.

```
X = df.drop('price' , axis = 1 )
y = df['price']
# Split data into combined training-validation set and test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Further split combined training-validation set into training and validation sets
train_size = 0.75  # 60% of the original data for training
val_size = 0.25    # 20% of the original data for validation

# Calculate the number of samples for training and validation sets
num_train_samples = int(len(X_train_val) * train_size)
num_val_samples = int(len(X_train_val) * val_size)

# Split the combined training-validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                  train_size=train_size, 
                                                  random_state=20)

# Verify the sizes of each dataset
print("Training dataset size:", len(X_train))
print("Validation dataset size:", len(X_val))
print("Test dataset size:", len(X_test))
```
| Dataset    | Size |
|------------|------|
| Training   | 327  |
| Validation | 109  |
| Test       | 109  |

*Splitted dataset table *

- Testing different models to find the best fit.

These six models will be tested to find the best fit.
1. Logistic Regression
2. Random Forest
3. K-Nearest Neighbor
4. Naive Bayes (Gaussian)
5. Support Vector Machine
6. Gradient Boosting Classifier

```
# Train the logistic regression model using the combined training-validation set
logistic_model = LogisticRegression()
logistic_model.fit(X_train_val, y_train_val)

# Predict on the validation set
logistic_pred_val = logistic_model.predict(X_val)

# Calculate accuracy on the validation set
logistic_accuracy_val = accuracy_score(y_val, logistic_pred_val) * 100
print("Logistic Regression Validation Accuracy:", logistic_accuracy_val)

# Predict on the test set
logistic_pred_test = logistic_model.predict(X_test)

# Calculate accuracy on the test set
logistic_accuracy_test = accuracy_score(y_test, logistic_pred_test) * 100
print("Logistic Regression Test Accuracy:", logistic_accuracy_test)
```
| Logistic Regression | Accuracy       |
|---------------------|----------------|
| Validation Accuracy | 3.6697         |
| Test Accuracy       | 1.8349         |

*Logistic Regression Accuracy Table*

```
# Train the random forest model using the combined training-validation set
random_forest_model = RandomForestClassifier(n_estimators=120)
random_forest_model.fit(X_train_val, y_train_val)

# Predict on the validation set
random_forest_pred_val = random_forest_model.predict(X_val)

# Calculate accuracy on the validation set
random_forest_accuracy_val = accuracy_score(y_val, random_forest_pred_val) * 100
print("Random Forest Validation Accuracy:", random_forest_accuracy_val)

# Predict on the test set
random_forest_pred_test = random_forest_model.predict(X_test)

# Calculate accuracy on the test set
random_forest_accuracy_test = accuracy_score(y_test, random_forest_pred_test) * 100
print("Random Forest Test Accuracy:", random_forest_accuracy_test)
```
| Random Forest       | Accuracy        |
|---------------------|-----------------|
| Validation Accuracy | 98.1651         |
| Test Accuracy       | 4.5872          |

*Random Forest Accuracy Table*

```
# Train the KNN model using the combined training-validation set
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_val, y_train_val)

# Predict on the validation set
knn_pred_val = knn_model.predict(X_val)

# Calculate accuracy on the validation set
knn_accuracy_val = accuracy_score(y_val, knn_pred_val) * 100
print("KNN Validation Accuracy:", knn_accuracy_val)

# Predict on the test set
knn_pred_test = knn_model.predict(X_test)

# Calculate accuracy on the test set
knn_accuracy_test = accuracy_score(y_test, knn_pred_test) * 100
print("KNN Test Accuracy:", knn_accuracy_test)
```
| K-Nearest Neighbor  | Accuracy        |
|---------------------|-----------------|
| Validation Accuracy | 33.0275         |
| Test Accuracy       | 0.9174          |

*K-Nearest Neighbor Accuracy Table*

```
# Train the Gaussian Naive Bayes model using the combined training-validation set
naive_bayes_gaussian = GaussianNB()
naive_bayes_gaussian.fit(X_train_val, y_train_val)

# Predict on the validation set
nbg_pred_val = naive_bayes_gaussian.predict(X_val)

# Calculate accuracy on the validation set
nbg_accuracy_val = accuracy_score(y_val, nbg_pred_val) * 100
print("Naive Bayes (Gaussian) Validation Accuracy:", nbg_accuracy_val)

# Predict on the test set
nbg_pred_test = naive_bayes_gaussian.predict(X_test)

# Calculate accuracy on the test set
nbg_accuracy_test = accuracy_score(y_test, nbg_pred_test) * 100
print("Naive Bayes (Gaussian) Test Accuracy:", nbg_accuracy_test)
```
| Naive Bayes (Gaussian) | Accuracy        |
|------------------------|-----------------|
| Validation Accuracy    | 55.0459         |
| Test Accuracy          | 0.0             |

*Naive Bayes (Gaussian) Accuracy Table*

```
# Train the Support Vector Machine (SVM) model using the combined training-validation set
svc_model = SVC()
svc_model.fit(X_train_val, y_train_val)

# Predict on the validation set
svm_pred_val = svc_model.predict(X_val)

# Calculate accuracy on the validation set
svm_accuracy_val = accuracy_score(y_val, svm_pred_val) * 100
print("Support Vector Machine Validation Accuracy:", svm_accuracy_val)

# Predict on the test set
svm_pred_test = svc_model.predict(X_test)

# Calculate accuracy on the test set
svm_accuracy_test = accuracy_score(y_test, svm_pred_test) * 100
print("Support Vector Machine Test Accuracy:", svm_accuracy_test)
```
| Support Vector Machine | Accuracy        |
|------------------------|-----------------|
| Validation Accuracy    | 1.8349          |
| Test Accuracy          | 0.9174          |

*Support Vector Machine Accuracy Table*

```
# Train the Gradient Boosting Classifier (GBC) model using the combined training-validation set
gbc_model = GradientBoostingClassifier()
gbc_model.fit(X_train_val, y_train_val)

# Predict on the validation set
gbc_pred_val = gbc_model.predict(X_val)

# Calculate accuracy on the validation set
gbc_accuracy_val = accuracy_score(y_val, gbc_pred_val) * 100
print("Gradient Boosting Classifier Validation Accuracy:", gbc_accuracy_val)

# Predict on the test set
gbc_pred_test = gbc_model.predict(X_test)

# Calculate accuracy on the test set
gbc_accuracy_test = accuracy_score(y_test, gbc_pred_test) * 100
print("Gradient Boosting Classifier Test Accuracy:", gbc_accuracy_test)
```

| Gradient Boosting Classifier | Accuracy        |
|------------------------------|-----------------|
| Validation Accuracy          | 94.4954         |
| Test Accuracy                | 2.7523          |

*Gradient Boosting Classifier Accuracy Table*

- Plotting a bar graph to visualise the results from all six tested models
```
# Define lists to store model names and accuracies
model_name = ['logistic regression', 'random forest', 'K-Nearest Neighbor', 'Naive Base', 'support vector machine', 'Gradient Boosting Classifier']
model_val_accuracies = [logistic_accuracy_val, random_forest_accuracy_val, knn_accuracy_val, nbg_accuracy_val, svm_accuracy_val, gbc_accuracy_val]
model_test_accuracies = [logistic_accuracy_test, random_forest_accuracy_test, knn_accuracy_test, nbg_accuracy_test, svm_accuracy_test, gbc_accuracy_test]

# Create a DataFrame with model names and accuracies
model_df = pd.DataFrame({'Model': model_name, 'Validation Accuracy': model_val_accuracies, 'Test Accuracy' : model_test_accuracies})

# Sort the DataFrame by accuracy in descending order
model_df = model_df.reset_index()
model_df_sorted = model_df.sort_values(by=['Validation Accuracy', 'Test Accuracy'], ascending=[False, False])

# Display the sorted DataFrame
print(model_df_sorted)

fig =px.bar(model_df_sorted, x=['Validation Accuracy', 'Test Accuracy'], y='Model',
           title= 'Validation and Test Accuracy levels for Each Model',
          labels={'value':'Accuracy Level','Model':'Model'},)
fig.update_layout(barmode='group')
fig.show()
```
![Bar Graph of Accuracy Test](/images/newplot.png)
*Bar graph visualizing validation and test accuracy of the six models*

The results shows that random forest model is the best fit model since it recorded the hightest values in terms of validation and test accuracy. 

- Using the model to predict house prices

1. Saving best model to file using pickle
```
#saving best model to file using pickle

file_name = "Price_Prediction_model.h5"

Price_Prediction_model = pickle.dump(random_forest_model,open(file_name, 'wb'))
```
2. The feature variables for price prediction
```
X_train.columns
```
Index(['area', 'bedrooms', 'bathrooms', 'stories', 'main_road', 'guestroom',
       'basement', 'hotwater_heating', 'air_conditioning', 'parking',
       'pref_area', 'furnishing_status'],
      dtype='object')


3. loading saved model for prediction
```
#loading saved model for prediction
model_filename = "Price_Prediction_model.h5"

Price_Prediction_model = pickle.load(open(model_filename, 'rb'))
#'area':any number, 'bedrooms':1-6 , 'bathrooms':1-4, 'stories':1-4, 'main_road':0,1, 'guestroom':0,1,
#'basement':0,1, 'hotwater_heating':0,1, 'air_conditioning':0,1, 'parking':0-3,
#'pref_area':0,1, 'furnishing_status':0,1,2

#sample from housing.csv, excel(spreadsheet) row 2
values=[7420,4,2,3,1,0,0,0,1,2,1,2]
df=pd.DataFrame([values])
df.columns=X_train.columns
#df
result = Price_Prediction_model.predict(df)

result[0]
```
The predicted price of $13,300,000 is exactly as what is in the dataset. This further shows that ramdon forest model is the best fit model. 

4. To deploy the app on Streamlit via GitHub, it is necessary to reduce the original file size to comply with GitHub’s size restrictions. When reducing the file size of an app or dataset to fit within GitHub's size limits, several limitations and trade-offs should be considered:
- Quality Loss: Compression may degrade data quality or model performance, affecting prediction accuracy.
- Increased Complexity: Managing smaller files can complicate data handling and preprocessing.
- Performance Impact: Compression adds overhead, potentially slowing down model loading.
- Dataset Reduction: Smaller datasets may lack diversity, limiting model generalization.
- Version Control Issues: Git LFS or external storage solutions may add complexity to your workflow.




**To run and deploy this machine learning model, run this py file [project_HousePricePredictionApp](/project_HousePricePredictionApp.py)**

**Visit the deployed app [here](https://predicthouseprices.streamlit.app/).**
