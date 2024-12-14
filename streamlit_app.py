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
data = pd.read_csv("https://raw.githubusercontent.com/ibniinggrianti/attritionprediction/refs/heads/master/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv")

with st.expander('Overall Statistics'):        
    if "Attrition" in data.columns:
        attrition_rate = data["Attrition"].value_counts()
          
        # Creating two columns for side-by-side plots
        col1, col2 = st.columns(2)
          
        # Bar Plot in the first column
        with col1:
            st.info("### Employee Attrition Counts")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette=["#1d7874", "#8B0000"], ax=ax)
            #ax.set_title("Employee Attrition Counts", fontweight="black", size=20, pad=20)
          
            # Adding value annotations to the bars
            for i, v in enumerate(attrition_rate.values):
                ax.text(i, v, v, ha="center", fontweight='black', fontsize=10)
          
            st.pyplot(fig)
          
        # Pie Chart in the second column
        with col2:
            st.info("### Employee Attrition Rate")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                attrition_rate, 
                labels=["No", "Yes"], 
                autopct="%.2f%%", 
                textprops={"fontweight": "black", "size": 18},
                colors=["#1d7874", "#AC1F29"],
                explode=[0, 0.1], 
                startangle=90
            )
            center_circle = plt.Circle((0, 0), 0.3, fc='white')
            fig.gca().add_artist(center_circle)
            #ax.set_title("Employee Attrition Rate", fontweight="black", size=20, pad=10)
          
            st.pyplot(fig)
          
    else:
        st.info("Please upload a CSV file to start the analysis.")

with st.expander('Statistics by Personal Data'):
    if "Age" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employee Distribution by Age
        with col1:
            st.info("### Employee Distribution by Age")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.histplot(x="Age", hue="Attrition", data=data, kde=True, palette=["#11264e", "#6faea4"], ax=ax)
            #ax.set_title("Employee Distribution by Age", fontweight="black", size=20, pad=10)
            st.pyplot(fig)

        # Visualization for Employee Distribution by Age & Attrition
        with col2:
            st.info("### Employee Distribution by Age & Attrition")
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.boxplot(x="Attrition", y="Age", data=data, palette=["#D4A1E7", "#6faea4"], ax=ax)
            #ax.set_title("Employee Distribution by Age & Attrition", fontweight="black", size=20, pad=10)
            st.pyplot(fig)
    
    else:
        st.info("Please upload a CSV file to start the analysis.")
    if "Gender" in data.columns and "Attrition" in data.columns:
        gender_attrition = data["Gender"].value_counts()

        # Creating two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Pie chart for gender distribution in the first column
        with col1:
            st.info("### Employees Distribution by Gender")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                gender_attrition, 
                autopct="%.0f%%", 
                labels=gender_attrition.index, 
                textprops={"fontweight": "black", "size": 20},
                explode=[0, 0.1], 
                startangle=90,
                colors=["#ffb563", "#FFC0CB"]
            )
            st.pyplot(fig)

        # Bar plot for attrition rate by gender in the second column
        with col2:
            st.info("### Employee Attrition Rate by Gender")
            new_df = data[data["Attrition"] == "Yes"]
            value_1 = data["Gender"].value_counts()
            value_2 = new_df["Gender"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.barplot(x=value_2.index, y=value_2.values, palette=["#D4A1E7", "#E7A1A1"], ax=ax)
            for index, value in enumerate(value_2):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", 
                        size=10, fontweight="black")
            st.pyplot(fig)
          
    else:
        st.info("Please upload a CSV file to start the analysis.")
    if "MaritalStatus" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Marital Status (Pie chart)
        with col1:
            st.info("### Employees by Marital Status")
            fig, ax = plt.subplots(figsize=(6, 6))
            value_1 = data["MaritalStatus"].value_counts()
            ax.pie(
                value_1.values,
                labels=value_1.index,
                autopct="%.1f%%",
                pctdistance=0.75,
                startangle=90,
                colors=['#E84040', '#E96060', '#E88181', '#E7A1A1'],
                textprops={"fontweight": "black", "size": 15}
            )
            # Add a white circle at the center to make it a donut chart
            center_circle = plt.Circle((0, 0), 0.4, fc='white')
            fig.gca().add_artist(center_circle)
            #ax.set_title("Employees by Marital Status", fontweight="black", size=20, pad=20)
            st.pyplot(fig)

        # Visualization for Attrition Rate by Marital Status (Bar plot)
        with col2:
            st.info("### Attrition Rate by Marital Status")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["MaritalStatus"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=value_2.index, y=value_2.values, palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7", "#E7A1A1"], ax=ax)
            #ax.set_title("Attrition Rate by Marital Status", fontweight="black", size=20, pad=20)

            # Add text annotations for each bar
            for index, value in enumerate(value_2):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", size=10, fontweight="black")
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to start the analysis.")
      
    if "Education" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Education (Bar Plot)
        with col1:
            st.info("### Employees Distribution by Education")
            value_1 = data["Education"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_1.index, 
                y=value_1.values, 
                order=value_1.index, 
                palette=["#FFA07A", "#D4A1E7", "#FFC0CB", "#87CEFA"], 
                ax=ax
            )
            #ax.set_title("Employees Distribution by Education", fontweight="black", size=20, pad=15)
            # Add value annotations on the bars
            for index, value in enumerate(value_1.values):
                ax.text(index, value, value, ha="center", va="bottom", fontweight="black", size=10)
            st.pyplot(fig)

        # Visualization for Employee Attrition by Education (Bar Plot)
        with col2:
            st.info("### Employee Attrition by Education")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["Education"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_2.index, 
                y=value_2.values, 
                order=value_2.index, 
                palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7", "#E7A1A1"], 
                ax=ax
            )
            #ax.set_title("Employee Attrition by Education", fontweight="black", size=18, pad=15)
            # Add value and percentage annotations on the bars
            for index, value in enumerate(value_2.values):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", 
                        ha="center", va="bottom", fontweight="black", size=10)
            st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to start the analysis.")
    if "EducationField" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Education Field (Bar Plot)
        with col1:
            st.info("### Employees by Education Field")
            value_1 = data["EducationField"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_1.index, 
                y=value_1.values, 
                order=value_1.index, 
                palette=["#FFA07A", "#D4A1E7", "#FFC0CB", "#87CEFA"], 
                ax=ax
            )
            #ax.set_title("Employees by Education Field", fontweight="black", size=20, pad=15)
            # Add value annotations on the bars
            for index, value in enumerate(value_1.values):
                ax.text(index, value, value, ha="center", va="bottom", fontweight="black", size=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)

        # Visualization for Employee Attrition by Education Field (Bar Plot)
        with col2:
            st.info("### Employee Attrition by Education Field")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["EducationField"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_2.index, 
                y=value_2.values, 
                order=value_2.index, 
                palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7"], 
                ax=ax
            )
            #ax.set_title("Employee Attrition by Education Field", fontweight="black", size=18, pad=15)
            # Add value and percentage annotations on the bars
            for index, value in enumerate(value_2.values):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", 
                        ha="center", va="bottom", fontweight="black", size=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to start the analysis.")
      
with st.expander('Statistics by Employee Detail'):
  if "Department" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Department (Bar Plot)
        with col1:
            st.info("### Employees by Department")
            value_1 = data["Department"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_1.index, 
                y=value_1.values, 
                palette=["#FFA07A", "#D4A1E7", "#FFC0CB"], 
                ax=ax
            )
            #ax.set_title("Employees by Department", fontweight="black", size=20, pad=20)
            # Add value annotations on the bars
            for index, value in enumerate(value_1.values):
                ax.text(index, value, str(value), ha="center", va="bottom", fontweight="black", size=10)
            st.pyplot(fig)

        # Visualization for Employee Attrition Rate by Department (Bar Plot)
        with col2:
            st.info("### Attrition Rate by Department")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["Department"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_2.index, 
                y=value_2.values, 
                palette=["#11264e", "#6faea4", "#FEE08B"], 
                ax=ax
            )
            #ax.set_title("Attrition Rate by Department", fontweight="black", size=20, pad=20)
            # Add value and percentage annotations on the bars
            for index, value in enumerate(value_2):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", 
                        fontweight="black", size=10)
            st.pyplot(fig)
  else:
       st.info("Please upload a CSV file to start the analysis.")
  if "JobRole" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Job Role (Bar Plot)
        with col1:
            st.info("### Employees by Job Role")
            value_1 = data["JobRole"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_1.index.tolist(), 
                y=value_1.values, 
                palette=["#FFA07A", "#D4A1E7", "#FFC0CB", "#87CEFA"],
                ax=ax
            )
            #ax.set_title("Employees by Job Role", fontweight="black", pad=15, size=18)
            ax.set_xticklabels(value_1.index, rotation=90)
            # Add value annotations on the bars
            for index, value in enumerate(value_1.values):
                ax.text(index, value, value, ha="center", va="bottom", fontweight="black", size=10)
            st.pyplot(fig)

        # Visualization for Attrition Rate by Job Role (Bar Plot)
        with col2:
            st.info("### Attrition Rate by Job Role")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["JobRole"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_2.index.tolist(), 
                y=value_2.values, 
                palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7", "#E7A1A1"], 
                ax=ax
            )
            #ax.set_title("Employee Attrition Rate by Job Role", fontweight="black", pad=15, size=18)
            ax.set_xticklabels(value_2.index, rotation=90)
            # Add value and percentage annotations on the bars
            for index, value in enumerate(value_2.values):
                ax.text(
                    index, value, f"{value} ({int(attrition_rate[index])}%)", 
                    ha="center", va="bottom", fontweight="black", size=10
                )
            st.pyplot(fig)
  else:
      st.info("Please upload a dataset with 'JobRole' and 'Attrition' columns to view visualizations.")
  if "JobLevel" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Job Level (Pie Chart)
        with col1:
            st.info("### Employees by Job Level")
            value_1 = data["JobLevel"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(
                value_1.values, 
                labels=value_1.index, 
                autopct="%.1f%%", 
                pctdistance=0.8, 
                startangle=90, 
                colors=['#FF6D8C', '#FF8C94', '#FFAC9B', '#FFCBA4', "#FFD8B1"], 
                textprops={"fontweight": "black", "size": 10}
            )
            center_circle = plt.Circle((0, 0), 0.4, fc='white')
            plt.gca().add_artist(center_circle)
            #ax.set_title("Employees by Job Level", fontweight="black", size=16, pad=15)
            st.pyplot(fig)

        # Visualization for Attrition Rate by Job Level (Bar Plot)
        with col2:
            st.info("### Attrition Rate by Job Level")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["JobLevel"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x=value_2.index, 
                y=value_2.values, 
                palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7", "#E7A1A1"], 
                ax=ax
            )
            #ax.set_title("Attrition Rate by Job Level", fontweight="black", size=16, pad=15)
            # Add percentage annotations to bars
            for index, value in enumerate(value_2.values):
                ax.text(
                    index, value, 
                    f"{value} ({int(attrition_rate[index])}%)", 
                    ha="center", va="bottom", 
                    fontweight="black", size=10
                )
            st.pyplot(fig)

  else:
      st.info("Please upload a dataset containing 'JobLevel' and 'Attrition' columns to display visualizations.")
      
with st.expander('Statistics by Job'):
    if "BusinessTravel" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Visualization for Employees by Business Travel (Pie Chart)
        with col1:
            st.info("### Employees by Business Travel")
            fig, ax = plt.subplots(figsize=(10, 10))
            value_1 = data["BusinessTravel"].value_counts()
            ax.pie(
                value_1.values,
                labels=value_1.index,
                autopct="%.1f%%",
                pctdistance=0.75,
                startangle=90,
                colors=['#E84040', '#E96060', '#E88181'],
                textprops={"fontweight": "black", "size": 15}
            )
            # Add a white circle at the center to make it a donut chart
            center_circle = plt.Circle((0, 0), 0.4, fc='white')
            fig.gca().add_artist(center_circle)
            #ax.set_title("Employees by Business Travel", fontweight="black", size=20, pad=20)
            st.pyplot(fig)

        # Visualization for Attrition Rate by Business Travel (Bar Plot)
        with col2:
            st.info("### Attrition Rate by Business Travel")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["BusinessTravel"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=value_2.index, 
                y=value_2.values, 
                palette=["#11264e", "#6faea4", "#FEE08B"], 
                ax=ax
            )
           # ax.set_title("Attrition Rate by Business Travel", fontweight="black", size=20, pad=20)

            # Add text annotations for each bar
            for index, value in enumerate(value_2):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", 
                        size=10, fontweight="black")
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to start the analysis.")
    if "JobSatisfaction" in data.columns and "Attrition" in data.columns:
      # Create two columns for side-by-side plots
      col1, col2 = st.columns(2)
  
      # Visualization for Employees by Job Satisfaction (Pie chart)
      with col1:
          st.info("### Employees by Job Satisfaction")
          fig, ax = plt.subplots(figsize=(6, 6))
          value_1 = data["JobSatisfaction"].value_counts()
          ax.pie(
              value_1.values,
              labels=value_1.index,
              autopct="%.1f%%",
              pctdistance=0.8,
              startangle=90,
              colors=['#FFB300', '#FFC300', '#FFD700', '#FFFF00'],
              textprops={"fontweight": "black", "size": 15}
          )
          # Add a white circle at the center to make it a donut chart
          center_circle = plt.Circle((0, 0), 0.4, fc='white')
          fig.gca().add_artist(center_circle)
          st.pyplot(fig)
  
      # Visualization for Attrition Rate by Job Satisfaction (Bar plot)
      with col2:
          st.info("### Attrition Rate by Job Satisfaction")
          new_df = data[data["Attrition"] == "Yes"]
          value_2 = new_df["JobSatisfaction"].value_counts()
          attrition_rate = np.floor((value_2 / value_1) * 100).values
          fig, ax = plt.subplots(figsize=(5, 4))
          sns.barplot(x=value_2.index, y=value_2.values, order=value_2.index, 
                      palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7"], ax=ax)
          
          # Add text annotations for each bar
          for index, value in enumerate(value_2):
              ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", size=10, fontweight="black")
          
          st.pyplot(fig)
  
    else:
        st.info("Please upload a CSV file to start the analysis.") 
    if "OverTime" in data.columns and "Attrition" in data.columns:
            # Create two columns for side-by-side plots
            col1, col2 = st.columns(2)
        
            # Visualization for Employees by OverTime (Pie chart)
            with col1:
                st.info("### Employees by OverTime")
                fig, ax = plt.subplots(figsize=(3, 3))
                value_1 = data["OverTime"].value_counts()
                ax.pie(
                    value_1.values,
                    labels=value_1.index,
                    autopct="%.1f%%",
                    pctdistance=0.75,
                    startangle=90,
                    colors=["#ffb563", "#FFC0CB"],
                    textprops={"fontweight": "black", "size": 10}
                )
                # Add a white circle at the center to make it a donut chart
                center_circle = plt.Circle((0, 0), 0.4, fc='white')
                fig.gca().add_artist(center_circle)
                st.pyplot(fig)
        
            # Visualization for Attrition Rate by OverTime (Bar plot)
            with col2:
                st.info("### Attrition Rate by OverTime")
                new_df = data[data["Attrition"] == "Yes"]
                value_2 = new_df["OverTime"].value_counts()
                attrition_rate = np.floor((value_2 / value_1) * 100).values
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(x=value_2.index.tolist(), y=value_2.values, palette=["#D4A1E7", "#E7A1A1"], ax=ax)
                
                # Add text annotations for each bar
                for index, value in enumerate(value_2):
                    ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", size=10, fontweight="black")
                
                st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to start the analysis.")
    if "PerformanceRating" in data.columns and "Attrition" in data.columns:
            # Create two columns for side-by-side plots
            col1, col2 = st.columns(2)
        
            # Visualization for Employees by PerformanceRating (Pie chart)
            with col1:
                st.info("### Employees by PerformanceRating")
                fig, ax = plt.subplots(figsize=(4, 4))
                value_1 = data["PerformanceRating"].value_counts()
                ax.pie(
                    value_1.values,
                    labels=value_1.index,
                    autopct="%.1f%%",
                    pctdistance=0.75,
                    startangle=90,
                    colors=["#ffb563", "#FFC0CB"],
                    textprops={"fontweight": "black", "size": 10}
                )
                # Add a white circle at the center to make it a donut chart
                center_circle = plt.Circle((0, 0), 0.4, fc='white')
                fig.gca().add_artist(center_circle)
                st.pyplot(fig)
        
            # Visualization for Attrition Rate by PerformanceRating (Bar plot)
            with col2:
                st.info("### Attrition Rate by PerformanceRating")
                new_df = data[data["Attrition"] == "Yes"]
                value_2 = new_df["PerformanceRating"].value_counts()
                attrition_rate = np.floor((value_2 / value_1) * 100).values
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(x=value_2.index.tolist(), y=value_2.values, palette=["#D4A1E7", "#E7A1A1"], ax=ax)
                
                # Add text annotations for each bar
                for index, value in enumerate(value_2):
                    ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", size=10, fontweight="black")
                
                st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to start the analysis.")

    if "WorkLifeBalance" in data.columns and "Attrition" in data.columns:
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)
    
        # Visualization for Employees by WorkLifeBalance (Pie chart)
        with col1:
            st.info("### Employees by WorkLifeBalance")
            fig, ax = plt.subplots(figsize=(6, 6))
            value_1 = data["WorkLifeBalance"].value_counts()
            ax.pie(
                value_1.values,
                labels=value_1.index,
                autopct="%.1f%%",
                pctdistance=0.75,
                startangle=90,
                colors=['#FF8000', '#FF9933', '#FFB366', '#FFCC99'],
                textprops={"fontweight": "black", "size": 15}
            )
            # Add a white circle at the center to make it a donut chart
            center_circle = plt.Circle((0, 0), 0.4, fc='white')
            fig.gca().add_artist(center_circle)
            st.pyplot(fig)
    
        # Visualization for Attrition Rate by WorkLifeBalance (Bar plot)
        with col2:
            st.info("### Attrition Rate by WorkLifeBalance")
            new_df = data[data["Attrition"] == "Yes"]
            value_2 = new_df["WorkLifeBalance"].value_counts()
            attrition_rate = np.floor((value_2 / value_1) * 100).values
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=value_2.index.tolist(), y=value_2.values, order=value_2.index, palette=["#11264e", "#6faea4", "#FEE08B", "#D4A1E7", "#E7A1A1"], ax=ax)
    
            # Add text annotations for each bar
            for index, value in enumerate(value_2.values):
                ax.text(index, value, f"{value} ({int(attrition_rate[index])}%)", ha="center", va="bottom", fontweight="black", size=10)
            
            st.pyplot(fig)
    
    else:
        st.info("Please upload a CSV file to start the analysis.")

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
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education = st.sidebar.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
job_level = st.sidebar.selectbox("Job Level", ["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
performance_rating = st.sidebar.selectbox("Performance Rating", ["Excellent", "Outstanding"])
work_life_balance = st.sidebar.selectbox("Work-Life Balance", ["Bad", "Good", "Best", "Better"])

# Create input feature vector for prediction
input_data = {
    "Age": [age],
    "Gender_Male": [1 if gender == "Male" else 0],
    "Gender_Female": [1 if gender == "Female" else 0],
    "MaritalStatus_Single": [1 if marital_status == "Single" else 0],
    "MaritalStatus_Married": [1 if marital_status == "Married" else 0],
    "MaritalStatus_Divorced": [1 if marital_status == "Divorced" else 0],
    "Education_Below College": [1 if education == "Below College" else 0],
    "Education_College": [1 if education == "College" else 0],
    "Education_Bachelor": [1 if education == "Bachelor" else 0],
    "Education_Master": [1 if education == "Master" else 0],
    "Education_Doctor": [1 if education == "Doctor" else 0],
    "EducationField_Life Sciences": [1 if education_field == "Life Science" else 0],
    "EducationField_Medical": [1 if education_field == "Medical" else 0],
    "EducationField_Marketing": [1 if education_field == "Marketing" else 0],
    "EducationField_Technical Degree": [1 if education_field == "Technical Degree" else 0],
    "EducationField_Human Resources": [1 if education_field == "Human Resources" else 0],
    "EducationField_Other": [1 if education_field == "Other" else 0],
    "Department_Research & Development": [1 if department == "Research & Development" else 0],
    "Department_Sales": [1 if department == "Sales" else 0],
    "Department_Human Resources": [1 if department == "Human Resources" else 0],
    "JobRole_Research Scientist": [1 if job_role == "Research Scientist" else 0],
    "JobRole_Sales Executive": [1 if job_role == "Sales Executive" else 0],
    "JobRole_Laboratory Technician": [1 if job_role == "Laboratory Technician" else 0],
    "JobRole_Manufacturing Director": [1 if job_role == "Manufacturing Director" else 0],
    "JobRole_Healthcare Representative": [1 if job_role == "Healthcare Representative" else 0],
    "JobRole_Manager": [1 if job_role == "Manager" else 0],
    "JobRole_Sales Representative": [1 if job_role == "Sales Representative" else 0],
    "JobRole_Research Director": [1 if job_role == "Research Director" else 0],
    "JobRole_Human Resources": [1 if job_role == "Human Resources" else 0],
    "JobLevel_Mid Level": [1 if job_level == "Mid Level" else 0],
    "JobLevel_Senior Level": [1 if job_level == "Senior Level" else 0],
    "JobLevel_Entry Level": [1 if job_level == "Entry Level" else 0],
    "JobLevel_Junior Level": [1 if job_level == "Junior Level" else 0],
    "JobLevel_Executive Level": [1 if job_level == "Executive Level" else 0],
    "BusinessTravel_Travel_Frequently": [1 if business_travel == "Travel_Frequently" else 0],
    "BusinessTravel_Non-Travel": [1 if business_travel == "Non-Travel" else 0],
    "JobSatisfaction_Low": [1 if job_satisfaction == "Low" else 0],
    "JobSatisfaction_High": [1 if job_satisfaction == "High" else 0],
    "JobSatisfaction_Medium": [1 if job_satisfaction == "Medium" else 0],
    "JobSatisfaction_Very High": [1 if job_satisfaction == "Very High" else 0],
    "OverTime_Yes": [1 if overtime == "Yes" else 0],
    "OverTime_No": [1 if overtime == "No" else 0],
    "PerformanceRating_Outstanding": [1 if performance_rating == "Outstanding" else 0],
    "PerformanceRating_Excellent": [1 if performance_rating == "Excellent" else 0],
    "WorkLifeBalance_Better": [1 if work_life_balance == "Better" else 0],
    "WorkLifeBalance_Best": [1 if work_life_balance == "Best" else 0],
    "WorkLifeBalance_Good": [1 if work_life_balance == "Good" else 0],
    "WorkLifeBalance_Bad": [1 if work_life_balance == "Bad" else 0],
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
