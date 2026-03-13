# Heart Disease Data Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())

# -----------------------------
# 2. Data Quality Check
# -----------------------------
print(df.shape)
print("Missing values:")
print(df.isnull().sum())
print("\nDuplicaties:", df.duplicated().sum())
print("\nData types:", df.dtypes)
print("\nAge:", df['age'].min(), "-", df['age'].max())
print("Cholesterol:", df['chol'].min(), "-", df['chol'].max())
print("Blood Pressure:", df['trestbps'].min(), "-", df['trestbps'].max())

# -----------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------
print(df.describe())
print(df['target'].value_counts())

# -----------------------------
# 4. Age Distribution
# -----------------------------
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# -----------------------------
# 5. Cholesterol Distribution
# -----------------------------
sns.boxplot(x='target', y='chol', data=df)
plt.title("Cholesterol vs Heart Disease")
plt.show()

# -----------------------------
# 6. Blood Pressure Distribution
# -----------------------------
sns.boxplot(x='target', y='trestbps', data=df)
plt.title("Blood Pressure vs Heart Disease")
plt.show()

# -----------------------------
# 7. Target Distribution
# -----------------------------
sns.countplot(x='target', data=df)
plt.title("Target Distribution (0=healthy, 1=disease)")
plt.show()

# -----------------------------
# 8. Heart Rate Analysis
# -----------------------------
sns.boxplot(x='target', y='thalach', data=df)
plt.title("Max Heart Rate vs Heart Disease")
plt.show()

# -----------------------------
# 9. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 10. Feature Correlation with Target
# -----------------------------
correlations = df.corr(numeric_only=True)['target'].sort_values(ascending=False)
print(correlations)

correlations.drop('target').head(10).plot(kind='bar', color='pink')
plt.title("Top 10 Features Correlated with Target")
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='pastel')
plt.title("Age vs Cholesterol by Target")
plt.show()

sns.countplot(x='sex', hue='target', data=df)
plt.title("Heart Disease by Gender (0 = Female, 1 = Male)")
plt.show()

sns.boxplot(x='target', y='age', data=df)
plt.title('Age vs Heart Disease')
plt.show()

# -----------------------------
# 11. Data Preprocessing
# -----------------------------
x = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -----------------------------
# 12. Handle Class Imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts().to_dict())

# -----------------------------
# 13. Train Machine Learning Model
# -----------------------------
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(x_train_smote, y_train_smote)

# -----------------------------
# 14. Model Evaluation
# -----------------------------
y_pred_smote = model_smote.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))

# -----------------------------
# 15. Confusion Matrix
# -----------------------------
sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 16. Feature Importance
# -----------------------------
importances = pd.Series(model_smote.feature_importances_, index=x.columns).sort_values(ascending=False)
print(importances)
importances.plot(kind='bar', color='pink')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()