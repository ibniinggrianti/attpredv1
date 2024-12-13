import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.title('Attrition Prediction')

st.subheader("Is your job worth keeping? Should you stay? Or just leave? Let's try!")
st.write("You can see below for more information")

# Load dataset (Ensure the CSV file is in the correct location)
df = pd.read_csv("https://raw.githubusercontent.com/ibniinggrianti/attritionpredicttest/refs/heads/master/editedIBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv")

X_raw = df.drop('Attrition', axis=1)

y_raw = df['Attrition']

with st.sidebar:
  st.header('Input Features')
  Age = st.slider("Age", min_value=18, max_value=60, value=25)
  st.write(f"Your selected age: {Age}.")

  Gender_options = ["Male", "Female"]
  selected_gender = st.pills("Gender", Gender_options, selection_mode="single")
  Gender = selected_gender[0] if selected_gender else None  # Handle empty selection
  st.markdown(f"Your selected gender: {selected_gender}.")

  MaritalStatus_options = ["Single", "Married", "Divorced"]
  selected_marital_status = st.pills("Marital Status", MaritalStatus_options, selection_mode="single")
  MaritalStatus = selected_marital_status[0] if selected_marital_status else None  # Handle empty selection
  st.markdown(f"Your selected Marital Status: {selected_marital_status}.")

# DataFrame for the input features
data = {
    'Age': [Age],
    'Gender': [Gender],
    'MaritalStatus': [MaritalStatus]
}
data['Gender'] = pd.Series(data['Gender']).map({'F': 'Female', 'M': 'Male'})
data['MaritalStatus'] = pd.Series(data['MaritalStatus']).map({'S': 'Single', 'M': 'Married', 'D': 'Divorced'})
input_df = pd.DataFrame(data, index=[0])
input_attrition = pd.concat([input_df, X_raw], axis=0)

# Display in Streamlit
with st.expander('Input Features'):
    st.write('**Input Features:**')
    st.dataframe(input_df)

    st.write('**Combined Attrition Data (Input + Original Dataset):**')
    st.dataframe(input_attrition)
  
#Encode
encode = ['Gender', 'MaritalStatus']
df_attrition = pd.get_dummies(input_attrition, columns=encode, prefix=encode)
#df_attrition[:1]
X = pd.get_dummies(df.drop('Attrition', axis=1), drop_first=True)  
X = df_attrition[1:]
input_row = df_attrition[:1]

# Encode y
target_mapper = {'Yes': 0,
                 'No': 1,}
#def target_encode(val):
  #return target_mapper[val]
#y = y_raw.map(target_mapper)
#y = y_raw.apply(target_encode)
y = df['Attrition'].map(target_mapper)


with st.expander('Data Preparation'):
  st.write('**Encoded X (Input Features)**')
  st.dataframe(input_row)
  st.write('**Encoded y**')
  st.dataframe(y)

# Train Random Forest model
clf = RandomForestClassifier(random_state=42)

# Step 1: Check for missing values
missing_data = X.isnull().sum()

# Drop rows with missing values in any column
X_clean = X_raw.copy()
X_clean = pd.get_dummies(X_clean, drop_first=True)
st.write(X_clean.isnull().sum())
X_clean = X.dropna()

# Drop columns with missing values
X_clean = X.dropna(axis=1)

# Step 3: Check for missing values again after handling
missing_data_after = X_clean.isnull().sum()
st.write("Missing Values After Cleaning:", missing_data_after)

y = y_raw.apply(target_encode)
st.write(y.isnull().sum())  # Check if there are missing values in y
st.write(X_clean.shape)
# Step 4: Train model with cleaned data
clf.fit(X_clean, y)

# Make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display results in Streamlit
with st.expander('Model Prediction'):
    st.write(f"**Prediction:** {'Attrition' if prediction[0] == 1 else 'No Attrition'}")
    st.write("**Prediction Probabilities:**")
    st.dataframe(pd.DataFrame(prediction_proba, columns=['No Attrition', 'Attrition']))
