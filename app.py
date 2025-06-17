import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

def plot_feature_importance(values, prediction):
    colors = ['green' if prediction==0 else 'red' for _ in values]
    plt.figure(figsize=(10,4))
    bars = plt.bar(feature_names, values, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Input Feature Values (Red = Heart Disease Risk)')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

st.title('Heart Disease Prediction Web App')

st.sidebar.header('Input Features')

def user_input_features():
    data = {}
    data['Age'] = st.sidebar.number_input('Age', min_value=1, max_value=120, value=49)
    data['Sex'] = st.sidebar.selectbox('Sex (0=Female,1=Male)', [0,1], index=1)
    data['ChestPainType'] = st.sidebar.selectbox('Chest Pain Type (0-3)', [0,1,2,3], index=2)
    data['RestingBP'] = st.sidebar.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=160)
    data['Cholesterol'] = st.sidebar.number_input('Cholesterol', min_value=50, max_value=600, value=180)
    data['FastingBS'] = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (0=No,1=Yes)', [0,1], index=0)
    data['RestingECG'] = st.sidebar.selectbox('Resting ECG (0-2)', [0,1,2], index=1)
    data['MaxHR'] = st.sidebar.number_input('Max Heart Rate', min_value=60, max_value=220, value=156)
    data['ExerciseAngina'] = st.sidebar.selectbox('Exercise Induced Angina (0=No,1=Yes)', [0,1], index=0)
    data['Oldpeak'] = st.sidebar.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    data['ST_Slope'] = st.sidebar.selectbox('ST Slope (0-2)', [0,1,2], index=1)
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Input features')
st.write(input_df)

if st.button('Predict'):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 0:
        st.success("✅ Person Not Having Heart Disease")
    else:
        st.error("❌ Person Having Heart Disease")

    plot_feature_importance(input_df.values[0], prediction)

st.markdown("---")
st.header("Or upload CSV for batch prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    data['Prediction'] = predictions
    data['Prediction'] = data['Prediction'].map({0: 'No Heart Disease', 1: 'Heart Disease'})
    st.write(data)
    csv = data.to_csv(index=False).encode()
    st.download_button(label="Download Predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')

# --------------------------
# ✅ Footer Section
# --------------------------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color:white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: balck;
        border-top: 1px solid #e0e0e0;
    }
    </style>

    <div class="footer">
        Prepared by <b>Alfaz Emon</b>
    </div>
""", unsafe_allow_html=True)
