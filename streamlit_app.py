import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title('Attrition Prediction')

st.subheader("Is your job worth keeping? Should you stay? Or just leave? Let's try!")
st.write("You can see below for more information")

# Load dataset (Ensure the CSV file is in the correct location)
df = pd.read_csv("https://raw.githubusercontent.com/ibniinggrianti/attritionpredicttest/refs/heads/master/editedIBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv")

# Data Preprocessing
# Encoding categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df_encoded.drop('Attrition_Yes', axis=1)  # Features
y = df_encoded['Attrition_Yes']  # Target (1 if attrition, 0 if no attrition)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy in Streamlit
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Sidebar for user input
st.sidebar.header("Input Features for Prediction")

age = st.sidebar.slider("Age", 18, 60, 30)
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources", "Technology"])
education = st.sidebar.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
job_level = st.sidebar.selectbox("Job Level", ["Junior Level", "Mid Level", "Senior Level"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Manager", "Technician"])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
performance_rating = st.sidebar.selectbox("Performance Rating", ["Excellent", "Outstanding", "Good", "Average"])
work_life_balance = st.sidebar.selectbox("Work-Life Balance", ["Bad", "Better", "Excellent"])

# Create input feature vector for prediction
input_data = {
    "Age": [age],
    "BusinessTravel_Travel_Frequently": [1 if business_travel == "Travel_Frequently" else 0],
    "BusinessTravel_Travel_Rarely": [1 if business_travel == "Travel_Rarely" else 0],
    "Department_Research & Development": [1 if department == "Research & Development" else 0],
    "Department_Sales": [1 if department == "Sales" else 0],
    "Education_Below College": [1 if education == "Below College" else 0],
    "Education_College": [1 if education == "College" else 0],
    "Education_Bachelor": [1 if education == "Bachelor" else 0],
    "Education_Master": [1 if education == "Master" else 0],
    "Education_Doctor": [1 if education == "Doctor" else 0],
    "Gender_Male": [1 if gender == "Male" else 0],
    "JobLevel_Mid Level": [1 if job_level == "Mid Level" else 0],
    "JobLevel_Senior Level": [1 if job_level == "Senior Level" else 0],
    "JobRole_Research Scientist": [1 if job_role == "Research Scientist" else 0],
    "JobRole_Sales Executive": [1 if job_role == "Sales Executive" else 0],
    "JobSatisfaction_Medium": [1 if job_satisfaction == "Medium" else 0],
    "JobSatisfaction_Very High": [1 if job_satisfaction == "Very High" else 0],
    "MaritalStatus_Married": [1 if marital_status == "Married" else 0],
    "OverTime_Yes": [1 if overtime == "Yes" else 0],
    "PerformanceRating_Outstanding": [1 if performance_rating == "Outstanding" else 0],
    "PerformanceRating_Excellent": [1 if performance_rating == "Excellent" else 0],
    "WorkLifeBalance_Better": [1 if work_life_balance == "Better" else 0],
}

input_df = pd.DataFrame(input_data)

# Ensure the input data has the same columns as the training data
input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

# Now, make the prediction
prediction = clf.predict(input_df)

# Compare the columns of the training data and the input data
missing_cols = set(X_train.columns) - set(input_df.columns)
if missing_cols:
    st.write(f"Missing columns: {missing_cols}")
else:
    prediction = clf.predict(input_df)
  
# Show prediction result
#if prediction == 1:
    #st.write("Prediction: **Yes**, the employee is likely to leave.")
#else:
    #st.write("Prediction: **No**, the employee is likely to stay.")
# Example: 70% chance of attrition (Yes), 30% chance of staying (No)
probabilities = clf.predict_proba(input_df)[0]

# Create a DataFrame for the prediction
df_prediction_proba = pd.DataFrame({
    'Attrition (Yes)': [probabilities[1]],
    'Attrition (No)': [probabilities[0]]
})

# Display predicted attrition probabilities using progress bars
st.subheader('Predicted Attrition Probabilities')
st.dataframe(df_prediction_proba,
             column_config={
               'Attrition (Yes)': st.column_config.ProgressColumn(
                 'Attrition (Yes)',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Attrition (No)': st.column_config.ProgressColumn(
                 'Attrition (No)',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)

# Display the predicted result (Yes or No)
prediction_label = "Yes" if probabilities[1] > 0.5 else "No"
st.success(f"Prediction: The employee is likely to stay (Attrition: {prediction_label})")
