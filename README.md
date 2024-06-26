# DataAnalysis
# Heart Failure Clinical Records Analysis

## Overview

This Streamlit application provides an interactive analysis of a heart failure clinical records dataset. The main components of the application include:

### 1. Dataset Overview
- **Dataset**: The dataset used is the heart failure clinical records dataset.
- **Loading Data**: The data is loaded using `pandas.read_csv()` and the first few rows are displayed.
- **Summary Statistics**: Provides a statistical summary of the dataset using `pandas.DataFrame.describe()`.

### 2. Summary Statistics
- **Age**: Average age is 60.29 years, ranging from 40 to 95 years.
- **Anaemia**: Approximately 47.44% of patients have anaemia.
- **Creatinine Phosphokinase**: Mean level is 586.76 mcg/L, ranging from 23 to 7861 mcg/L.
- **Diabetes**: About 43.94% of patients have diabetes.
- **Ejection Fraction**: Average ejection fraction is 37.73%, ranging from 14% to 80%.
- **High Blood Pressure**: Approximately 36.48% of patients have high blood pressure.
- **Platelets**: Average platelet count is 265,075.40 kiloplatelets/mL.
- **Serum Creatinine**: Mean level is 1.37 mg/dL, with values ranging from 0.5 to 9.4 mg/dL.
- **Serum Sodium**: Average level is 136.81 mEq/L, ranging from 113 to 148 mEq/L.
- **Sex**: 64.56% of patients are male.
- **Smoking**: About 31.18% of patients smoke.
- **Time**: Follow-up period ranges from 4 to 285 days, with an average of 130.68 days.
- **Death Event**: 31.36% of patients died during the follow-up period.

### 3. Analysis
- **Missing Values**: No missing values in the dataset.
- **Data Types**: Includes both numerical and categorical variables.

### 4. Smoking Rate by Sex
- **Smoking Analysis**: Displays the smoking rate by sex, showing higher smoking rates among males compared to females.

### 5. Age Distribution
- **KDE Plot**: Kernel Density Estimate plot for age distribution, showing a slightly right-skewed distribution with a peak around 60 years.

### 6. Histograms of Key Variables
- **Histograms**: Provides histograms for key variables such as creatinine phosphokinase, ejection fraction, serum creatinine, serum sodium, and platelets, revealing their distributions.

### 7. Value Counts of Categorical Variables
- **Categorical Analysis**: Displays value counts for categorical variables like anaemia, diabetes, high blood pressure, sex, smoking, and death events.

### 8. Correlation Analysis
- **Correlation Heatmap**: Displays the correlation matrix using a heatmap, highlighting significant correlations between variables.

### 9. Distribution of Age by Death Event
- **Age and Death Event**: Analyzes the distribution of age by death event, showing higher death frequency among older patients.

### 10. Conclusion
- **Insights**: Summarizes key findings, such as demographics, health conditions, smoking rates, significant correlations, and distributions of health indicators.
- **Future Research**: Suggests further research and predictive modeling to identify high-risk patients and improve outcomes.

This application uses various Python libraries, including:
- **Streamlit**: For building the interactive web application.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For creating static visualizations.
- **Plotly**: For creating interactive visualizations.