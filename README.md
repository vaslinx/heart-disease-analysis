# Heart Disease

## About project
Exploratory data analysis and machine learning on a heart disease dataset.
Covers data cleaning, visualization, correlation analysis, SMOTE balancing, and RandomForest classification.

## Dataset
* **Source**: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — UCI Machine Learning Repository
* **Publisher**: UCI Machine Learning Repository
* **Description**: Medical data used to predict the presence of heart disease based on health indicators
* **Note**: The dataset used in this project is a synthetically generated sample based on the structure of the original UCI Heart Disease dataset. It was created for practice and learning purposes only.
* **License**: Public Domain

## Project structure
* heart.csv
* heart.ipynb 
* heart.py 

## Project Goals
* Data cleaning and quality check (missing values, duplicates, data types)
* Identifying factors associated with heart disease
* Finding relationships between health indicators
* Building visualizations (histograms, boxplots, heatmaps, countplots)
* Correlation analysis
* Handling class imbalance using SMOTE
* Machine learning analysis (RandomForest classification with model evaluation)

## Key Findings

**Age Distribution**: 
The sample includes participants aged 29 to 75. The distribution is relatively uniform with two noticeable peaks — around age 40-42 and 70-71, suggesting the sample covers both middle-aged and older adults fairly evenly.

**Target Distribution**:
The dataset is heavily imbalanced — patients with heart disease (~245) outnumber healthy cases (~60) by almost 4 to 1. This is an important limitation that may affect model performance.

**Age vs Heart Disease**:
Patients with heart disease tend to be older — median age for the disease group (~53) is higher than for healthy participants (~45). The wider box in the disease group (44-67) suggests greater age variability among those with heart disease.

**Correlation Heatmap**:
The correlation matrix shows that no single variable has a strong association with heart disease. The most notable weak positive correlations with target are: exercise-induced angina (r = 0.28), sex (r = 0.23), number of major vessels (r = 0.21), ST depression (r = 0.17), and resting blood pressure (r = 0.15). Thalassemia type shows a weak negative correlation (r = -0.12). The absence of a dominant predictor suggests that heart disease diagnosis requires a comprehensive approach — including data on physical activity, lifestyle, and habits.

**Heart Disease by Gender**:
Among males, the number of heart disease cases significantly exceeds healthy cases (~200 vs ~35). In the female group the ratio is more balanced (~40 vs ~25). This may reflect a real pattern — males are statistically more prone to heart disease — but may also be influenced by gender imbalance in the sample, which limits conclusions about the female group.

**Top 10 Features Correlated with Target**:
The chart confirms the correlation matrix findings -exercise-induced angina (0.28), sex (0.23), and number of major vessels (0.21) remain the strongest predictors. ST slope and thalassemia show weak negative correlations. Scatterplot

**Age vs Cholesterol**: 
Data points from both groups are evenly mixed across the plot with no clear separation — suggesting that neither age nor cholesterol alone is a reliable predictor of heart disease. The dominance of orange points reflects the overall dataset imbalance.

**Cholesterol vs Heart Disease**: 
Cholesterol levels are similar across both groups -median is slightly higher in the disease group (~330 vs ~300) with comparable spread. This confirms the earlier finding that cholesterol is not a key predictor of heart disease in this dataset.

**Max Heart Rate vs Heart Disease**: 
Median maximum heart rate is nearly identical in both groups (~138-140). However, healthy participants show slightly more values in the 110-135 range, while the disease group shows more values in the 135-162 range - the difference is minor and does not allow for a definitive conclusion.

**Confusion Matrix (SMOTE)**:
The model achieved ~80% accuracy (49 out of 61 predictions correct). It performs well at identifying patients with heart disease — 41 out of 44 actual cases were correctly classified. The main error is 9 healthy patients misclassified as diseased (False Positive). The most critical error is 3 diseased patients classified as healthy (False Negative) — in a medical context this is the most dangerous type of mistake. The use of SMOTE helped the model better identify the disease class despite the dataset imbalance.

**Feature Importance (RandomForest)**:
RandomForest identified exercise-induced angina (exang, 0.18), ST depression (oldpeak, 0.15), and resting blood pressure (trestbps, 0.12) as the most important features. Notably, sex — which ranked second in the correlation matrix — proved relatively unimportant for the model (0.047). This suggests that RandomForest captures more complex non-linear relationships between variables than simple correlation analysis.

## Conclusions
The analysis revealed a significant dataset imbalance — patients with heart disease (~245) outnumber healthy cases (~60) by almost 4 to 1, which may affect result quality. The correlation matrix showed no dominant predictor of heart disease — all correlations are weak. This suggests that clinical indicators alone are insufficient for accurate diagnosis — more comprehensive data on lifestyle, physical activity, and habits is needed. An additional limitation is the gender imbalance — males significantly outnumber females (~200 vs ~45), limiting conclusions about the female group. The RandomForest model achieved ~80% accuracy. However, since this dataset is synthetic and was generated for learning purposes, all conclusions are illustrative and cannot be applied in clinical practice.

## Limitations:
* The dataset is heavily imbalanced — ~245 disease cases vs ~60 healthy (~4:1), which may bias model predictions toward the disease group 
* The sample is gender-imbalanced - males significantly outnumber females (~200 vs ~45), limiting conclusions about the female group
* No data on lifestyle, physical activity, or habits - key factors for heart disease diagnosis
* The dataset is synthetic and generated for learning purposes only - conclusions cannot be applied in clinical practice
* RandomForest was used without hyperparameter tuning — model performance could be improved

## Next Steps
* Replace synthetic data with the real UCI Heart Disease dataset for clinically meaningful conclusions
* Tune RandomForest hyperparameters (max_depth, n_estimators) to improve model performance
* Add statistical testing (t-test) to verify differences between healthy and disease groups
* Try other classification models (Logistic Regression, XGBoost) and compare performance
* Add lifestyle and habit data to improve diagnostic accuracy

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
3. Open project.ipynb in Jupyter Notebook and run all cells

## Technologies
![Python](https://img.shields.io/badge/Python-blue)
![pandas](https://img.shields.io/badge/pandas-lightgrey)
![seaborn](https://img.shields.io/badge/seaborn-teal)
![matplotlib](https://img.shields.io/badge/matplotlib-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Author
[vaslinx] · [GitHub]( https://github.com/vaslinx)
