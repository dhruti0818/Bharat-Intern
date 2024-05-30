import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('housing.csv')
# Initial exploration of the dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# Handle missing values by removing rows with any null values
df.dropna(inplace=True)
print(df.isnull().sum())

# One-hot encode the 'ocean_proximity' categorical column
ocean_proximity_encoded = pd.get_dummies(df['ocean_proximity']).astype(int)
df = df.join(ocean_proximity_encoded)
df.drop(columns=['ocean_proximity'], inplace=True)

# One-hot encode the 'ocean_proximity' categorical column
ocean_proximity_encoded = pd.get_dummies(df['ocean_proximity']).astype(int)
df = df.join(ocean_proximity_encoded)
df.drop(columns=['ocean_proximity'], inplace=True)

# Apply log transformation to specific columns to normalize distribution
df['total_rooms'] = np.log1p(df['total_rooms'])
df['total_bedrooms'] = np.log1p(df['total_bedrooms'])
df['population'] = np.log1p(df['population'])
df['households'] = np.log1p(df['households'])

# Create new feature columns for better model performance
df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
df['household_rooms'] = df['total_rooms'] / df['households']
# Create new feature columns for better model performance
df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
df['household_rooms'] = df['total_rooms'] / df['households']
# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression Train Score:", linear_model.score(X_train, y_train))
print("Random Forest Regressor Test Score:", forest_model.score(X_test, y_test))
# Predict using the Random Forest model
sample_data = np.array([[-142.23, 37.10, 43.0, 5.781058, 6.8674563, 7.777652, 3.804177, 8.3252, 2, 1, 2, 4, 5, 0.867813, 1.322834]])
predicted_value = forest_model.predict(sample_data)
print("Predicted Median House Value:", predicted_value)
