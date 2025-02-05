# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files

# Step 1: Upload the CSV file
uploaded = files.upload()

# Step 2: Read the uploaded file
csv_filename = list(uploaded.keys())[0]  # Get the uploaded filename
df = pd.read_csv(csv_filename)

# Step 3: Display basic dataset info
print("Dataset Preview:")
print(df.head())

# Step 4: Check and Handle Missing Values
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df.fillna(df.median(), inplace=True)  # Replace missing values with median
else:
    print("\nNo missing values found.")

# Step 5: Separate Features and Target Variable
X = df.drop(columns=['fail'])
y = df['fail']

# Step 6: Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Normalize Features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train a RandomForest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 10: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 11: Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 12: Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.show()
