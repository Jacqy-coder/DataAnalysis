import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy

# Load the dataset
file_path = 'heart_failure_clinical_records.csv'  # Assuming you downloaded the file and placed it in the same directory
df = pd.read_csv(file_path)

# App title
st.title('Heart Failure Clinical Records Analysis')

# Display dataset overview
st.header('Dataset Overview')
st.write(df.head())

# Display summary statistics
st.header('Summary Statistics')
st.write(df.describe())

# Summary Statistics
st.header('Summary Statistics')
st.write("""
- **Age**: The average age is 60.29 years, with a standard deviation of 11.70 years. The ages range from 40 to 95 years.
- **Anaemia**: Approximately 47.44% of the patients have anaemia.
- **Creatinine Phosphokinase**: The mean level is 586.76 mcg/L, with a wide range from 23 to 7861 mcg/L.
- **Diabetes**: About 43.94% of the patients have diabetes.
- **Ejection Fraction**: The average ejection fraction is 37.73%, ranging from 14% to 80%.
- **High Blood Pressure**: Approximately 36.48% of the patients have high blood pressure.
- **Platelets**: The average platelet count is 265,075.40 kiloplatelets/mL.
- **Serum Creatinine**: The mean level is 1.37 mg/dL, with values ranging from 0.5 to 9.4 mg/dL.
- **Serum Sodium**: The average level is 136.81 mEq/L, ranging from 113 to 148 mEq/L.
- **Sex**: 64.56% of the patients are male.
- **Smoking**: About 31.18% of the patients smoke.
- **Time**: The follow-up period ranges from 4 to 285 days, with an average of 130.68 days.
- **Death Event**: 31.36% of the patients died during the follow-up period.
""")

# Analysis after summary and missing values
st.subheader('Analysis')
st.write("""
- **Missing Values**: The dataset contains no missing values.
- **Summary Statistics**: Provides descriptive statistics for numerical variables.
- **Data Types**: The dataset includes both numerical and categorical variables.
""")

# Smoking rate by sex
st.header('Smoking Rate by Sex')
smoke_rate = df.groupby(['sex'])['smoking'].sum().reset_index()
st.write(smoke_rate)

# Analysis of smoking rate by sex
st.subheader('Analysis')
st.write("""
The smoking rate is higher among males (1006) compared to females (553).
""")

# KDE plot of age
st.header('Age Distribution')
fig, ax = plt.subplots()
df['age'].plot(kind='kde', title='KDE of Age', ax=ax)
plt.xlabel('Age')
plt.ylabel('Density')
st.pyplot(fig)

# Analysis of age distribution
st.subheader('Analysis')
st.write("""
The KDE plot shows a slightly right-skewed age distribution, with a peak around 60 years.
""")

# Histograms of key variables
st.header('Histograms of Key Variables')
key_vars = ['creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'platelets']
fig, axs = plt.subplots(len(key_vars), 1, figsize=(10, 20))
for i, var in enumerate(key_vars):
    axs[i].hist(df[var], bins=30)
    axs[i].set_title(f'Histogram of {var}')
    axs[i].set_xlabel(var)
    axs[i].set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig)

# Analysis of histograms
st.subheader('Analysis')
st.write("""
Histograms reveal distributions:
- **Creatinine Phosphokinase**: Highly skewed with extreme values.
- **Ejection Fraction**: Fairly uniform distribution.
- **Serum Creatinine and Sodium**: Normal distribution with slight skewness.
- **Platelets**: Normal distribution with outliers.
""")

# Value counts of categorical variables
st.header('Value Counts of Categorical Variables')
count_dict = {
    'anaemia': df['anaemia'].value_counts(),
    'diabetes': df['diabetes'].value_counts(), 
    'high_blood_pressure': df['high_blood_pressure'].value_counts(),
    'sex': df['sex'].value_counts(),
    'smoking': df['smoking'].value_counts(),
    'DEATH_EVENT': df['DEATH_EVENT'].value_counts()
}
count_df = pd.DataFrame.from_dict(count_dict)
st.write(count_df)

# Analysis of value counts
st.subheader('Analysis')
st.write("""
Counts of categorical variables:
- **Anaemia**: 52.56% no, 47.44% yes.
- **Diabetes**: 56.06% no, 43.94% yes.
- **High Blood Pressure**: 63.52% no, 36.48% yes.
- **Sex**: 35.44% female, 64.56% male.
- **Smoking**: 68.82% no, 31.18% yes.
- **Death Event**: 68.64% no, 31.36% yes.
""")

# Correlation analysis
st.header('Correlation Analysis')
corr_matrix = df.corr()
fig = px.imshow(corr_matrix, labels=dict(color="Correlation"), x=corr_matrix.index, y=corr_matrix.columns, color_continuous_scale='RdBu', width=800, height=600)
fig.update_layout(title='Correlation Heatmap')
st.plotly_chart(fig)

# Analysis of correlation
st.subheader('Analysis')
st.write("""
- **Age** is negatively correlated with **ejection fraction** and **platelets**.
- **Ejection Fraction** is negatively correlated with **serum creatinine** and **creatinine phosphokinase**.
- **DEATH_EVENT** is positively correlated with **serum creatinine** and negatively with **ejection fraction**.
""")

# Distribution of age by death event
st.header('Distribution of Age by Death Event')
melted_df = df.melt(id_vars='DEATH_EVENT', value_vars=['age'], var_name='Factor', value_name='Value')
fig, ax = plt.subplots()
sns.histplot(data=melted_df, x='Value', hue='DEATH_EVENT', multiple='stack', palette='Set1', kde=False, ax=ax)
plt.title('Distribution of Age')
plt.xlabel('Value')
plt.ylabel('Frequency')
st.pyplot(fig)

# Analysis of age by death event
st.subheader('Analysis')
st.write("""
The age distribution is similar between those who survived and those who did not, with higher death frequency among older patients.
""")

# Conclusion
st.header('Conclusion')
st.write("""
This analysis of the heart failure clinical records dataset reveals several insights:

- **Demographics and Health Conditions**: The dataset consists mainly of older patients, with a significant portion suffering from conditions like anaemia, diabetes, and high blood pressure.
- **Smoking**: More male patients smoke compared to female patients.
- **Key Correlations**: Significant correlations exist between age, ejection fraction, serum creatinine, and the occurrence of death events. Higher serum creatinine and lower ejection fraction are associated with higher mortality.
- **Distributions**: Various health indicators exhibit diverse distributions, with some like creatinine phosphokinase showing high skewness, indicating outliers or extreme values.

These findings provide a foundation for further research, including predictive modeling to identify high-risk patients and targeted interventions to improve patient outcomes.
""")
