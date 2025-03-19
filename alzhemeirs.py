from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

file_path = "alzheimers_disease_data.csv"
df = pd.read_csv(file_path)

# Drop the rows with missing values
df = df.dropna()
df = df.drop("PatientID", axis=1)

# Removing Extra white spaces
df.columns = df.columns.str.strip()
df.head()
df.info()
df.describe()
missing_values = df.isna().mean() * 100
missing_values.sum()
duplicated_values = df.duplicated().sum()
duplicated_values
# Checking the distribution of the target variable
alzheimers_distribution = df['Diagnosis'].value_counts()
alzheimers_distribution

numerical_df = df.select_dtypes(include=["int64", "float64"])

# Compute and show the correlation matrix
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title("Correlation Matrix of Alzheimer's Disease Dataset", fontsize=20)
plt.show()

X = df.drop(columns=["DoctorInCharge", "Diagnosis", "Forgetfulness", "DifficultyCompletingTasks",
                      "Hypertension", "Confusion", "HeadInjury", "Depression", "FamilyHistoryAlzheimers",
                      "CardiovascularDisease"])
Y = df["Diagnosis"]

# Scale Fit Transformation
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Set up the model
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                   n_iter=50, scoring='accuracy', cv=3, verbose=1, 
                                   random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters and the best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)

# Use the best model to make predictions
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importances = best_rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances) 
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

import joblib

# Save the model
joblib.dump(rf_model, "alzheimers_model.pkl")