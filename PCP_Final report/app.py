import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load Trained Model
model = joblib.load("LogisticRegression_model.pkl")

# Automatically Load CSV File
csv_file_path = "heart_disease_high_accuracy.csv"  # Change this path if needed

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Splitting Data
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    # Predictions
    y_pred = model.predict(X)

    # Compute Classification Metrics
    report = classification_report(y, y_pred, output_dict=True)
    accuracy = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Display Model Performance
    st.write("## Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1_score:.2%}")

    # Display Classification Report
    st.write("### Classification Report")
    df_report = pd.DataFrame(report).T.round(2)
    st.dataframe(df_report)

    # Uncomment below if you'd like to display confusion matrix heatmap
    # st.write("### Confusion Matrix")
    # fig, ax = plt.subplots(figsize=(5, 4))
    # sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix (Percentage)")
    # st.pyplot(fig)

else:
    st.error(f"Dataset not found! Please place the CSV file at `{csv_file_path}`.")

# ----------------------------------------
# User Input for Heart Disease Prediction
# ----------------------------------------

st.write("## Enter Your Health Details to Check Coronary Heart Disease Risk")

# User Inputs with help text on the left
st.markdown("ℹ️ Your age in years. This is important because heart disease risk increases with age.")
age = st.number_input("Age", min_value=20, max_value=100, value=50)

st.markdown("ℹ️ Choose your biological sex. Men and women can have different heart disease risks.")
sex = st.radio("Sex", ["Male", "Female"])

st.markdown("ℹ️ Describes the type of chest pain you may feel. 0=No pain, 1=Mild pain, 2=Moderate pain, 3=Severe pain, and so on.")
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])

st.markdown("ℹ️ Your blood pressure when you are resting. Normal is around 120 mm Hg. High blood pressure increases heart risk.")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

st.markdown("ℹ️ The amount of cholesterol (fat) in your blood. High levels can block arteries. Healthy level is below 200.")
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

st.markdown("ℹ️ Is your blood sugar level higher than 120 mg/dL after not eating for 8 hours? High sugar can increase heart risk.")
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])

st.markdown("ℹ️ Result from a heart test (ECG) when you're resting. 0=Normal, 1=Possible issue, 2=May show heart strain.")
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

st.markdown("ℹ️ The highest heart rate you reached during exercise. Shows how well your heart handles activity.")
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

st.markdown("ℹ️ Do you feel chest pain when doing physical activity? This could be a sign of reduced blood flow to your heart.")
exang = st.radio("Exercise-Induced Angina", ["No", "Yes"])

st.markdown("ℹ️ Measures changes in your heart activity after exercise. Higher values may indicate heart problems.")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1, value=1.0)

st.markdown("ℹ️ The shape of your heart's activity line during exercise. 0=Rising, 1=Flat, 2=Falling. Flat or falling may be risky.")
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])

st.markdown("ℹ️ The number of major heart vessels that can be seen in an X-ray scan. More visible vessels usually means better health.")
ca = st.selectbox("Number of Major Vessels (0-4) Colored by Fluoroscopy", [0, 1, 2, 3, 4])

st.markdown("ℹ️ A blood disorder affecting red blood cells. 0=Unknown, 1=Normal, 2=Permanent defect, 3=Defect that may come and go.")
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Create feature array
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction Button
if st.button("Check Heart Disease Risk"):
    prediction = model.predict(input_features)
    result = "No Heart Disease" if prediction[0] == 0 else "Heart Disease Detected"

    # Display Result
    st.write("## Prediction Result")
    st.write(f"### 🏥 {result}")

    if prediction[0] == 0:
        st.success("✅ No Heart Disease Detected. Keep maintaining a healthy lifestyle!.")
    else:
        st.error("⚠️ High Risk of Heart Disease. Please consult a doctor")

