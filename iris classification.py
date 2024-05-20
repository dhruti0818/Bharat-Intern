import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset and store it in a Pandas DataFrame
iris_data = load_iris()
df_iris = pd.DataFrame(data= np.c_[iris_data['data'], iris_data['target']], columns= iris_data['feature_names'] + ['target'])

# Visualize the data using boxplots to check for outliers
sns.set(style="whitegrid")
plt.figure(figsize=(10, 7))
sns.boxplot(data=df_iris)
plt.title('Feature Distribution')
plt.show()

# Check the data types of the DataFrame
print("Data types:\n", df_iris.dtypes)

# Display the shape of the DataFrame
print("\nShape of the DataFrame:\n", df_iris.shape)

# Calculate the correlation matrix
corr_matrix = df_iris.corr()
print("\nCorrelation matrix:\n", corr_matrix)

# Calculate skewness for each column
skewness = df_iris.skew()
print("\nSkewness:\n", skewness)

# Check unique values in the target column
print("\nUnique values in the target column:\n", df_iris['target'].value_counts())

# Split the data into features and target
X = df_iris.drop('target', axis=1)
y = df_iris['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
