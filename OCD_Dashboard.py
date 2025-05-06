import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("OCD_Patient_Dataset_Demographics_&_Clinical_Data.csv")
df["Total Y-BOCS Score"] = df["Y-BOCS Score (Obsessions)"] + df["Y-BOCS Score (Compulsions)"]
df["OCD Diagnosis"] = df["Total Y-BOCS Score"].apply(lambda x: 1 if x >= 16 else 0)

# Configure Streamlit page
st.set_page_config(page_title="OCD Patient Analysis Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🧠 OCD Patient Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>An interactive tool to analyze, visualize, and predict OCD using clinical data.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Insights", "📈 Visual Analysis", "🤖 OCD Prediction Model"])

# Top KPIs
total_patients = df.shape[0]
ocd_positive = df["OCD Diagnosis"].sum()
average_age = int(df["Age"].mean())

st.markdown("### 📌 Key Patient Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", total_patients)
col2.metric("OCD Diagnosed", ocd_positive)
col3.metric("Average Age", average_age)

st.markdown("---")

# Page: Data Insights
if page == "📊 Data Insights":
    st.header("🧬 Demographic Overview")
    gender_counts = df['Gender'].value_counts()
    ethnicity_counts = df['Ethnicity'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="Blues_d", ax=ax1)
        ax1.set_title("Gender Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.barplot(x=ethnicity_counts.index, y=ethnicity_counts.values, palette="Greens_d", ax=ax2)
        ax2.set_title("Ethnicity Distribution")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("📅 Monthly OCD Diagnosis Trends")
    df['OCD Diagnosis Date'] = pd.to_datetime(df['OCD Diagnosis Date'])
    df['Month'] = df['OCD Diagnosis Date'].dt.strftime('%B')
    month_counts = df['Month'].value_counts().sort_index()

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=month_counts.index, y=month_counts.values, palette="Oranges", ax=ax3)
    ax3.set_title("Patients Diagnosed per Month")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

# Page: Visual Analysis
elif page == "📈 Visual Analysis":
    st.header("💊 Treatment & Symptom Severity")

    st.subheader("Treatment Methods")
    treatment_counts = df['Medications'].value_counts()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=treatment_counts.index, y=treatment_counts.values, palette="coolwarm", ax=ax4)
    ax4.set_title("Treatment Methods Used")
    st.pyplot(fig4)

    st.subheader("Symptom Severity by Gender")
    col1, col2 = st.columns(2)

    with col1:
        fig5 = plt.figure()
        sns.boxplot(x="Gender", y="Y-BOCS Score (Obsessions)", data=df, palette="pastel")
        plt.title("Obsessions Severity by Gender")
        st.pyplot(fig5)

    with col2:
        fig6 = plt.figure()
        sns.boxplot(x="Gender", y="Y-BOCS Score (Compulsions)", data=df, palette="muted")
        plt.title("Compulsions Severity by Gender")
        st.pyplot(fig6)

# Page: Prediction Model
elif page == "🤖 OCD Prediction Model":
    st.header("🔍 Predict OCD Diagnosis")

    df_clean = df.dropna(subset=['Age', 'Duration of Symptoms (months)', 'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)'])
    X = df_clean[['Age', 'Duration of Symptoms (months)', 'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)']]
    y = df_clean['OCD Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "ocd_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        duration = st.number_input("Duration of Symptoms (months)", min_value=0, max_value=120, value=12)
    with col2:
        obs_score = st.slider("Y-BOCS Obsessions Score", 0, 40, 20)
        comp_score = st.slider("Y-BOCS Compulsions Score", 0, 40, 18)

    input_df = pd.DataFrame([[age, duration, obs_score, comp_score]],
                            columns=['Age', 'Duration of Symptoms (months)', 'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)'])

    model = joblib.load("ocd_model.pkl")
    scaler = joblib.load("scaler.pkl")

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    if st.button("🔮 Predict OCD"):
        result = "🟢 OCD Positive" if prediction[0] == 1 else "🔵 OCD Negative"
        st.success(f"Prediction: {result}")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Created with ❤️ using Streamlit | Project by Prajakta </p>", unsafe_allow_html=True)
