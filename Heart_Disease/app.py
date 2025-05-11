import streamlit as st
import pickle
import pandas as pd
import numpy as np
import base64
import sklearn

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model information'])

with tab1:
    # Inputs
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])  
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120mg/dl", ">120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    old_peak = st.number_input("Old Peak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Encoding inputs for model compatibility
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create DataFrame for input data with matching column names
    input_data = pd.DataFrame(
        {
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [old_peak],
            'ST_Slope': [st_slope]   # ✅ Corrected key
        })


    # Model names and algorithm names
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl']

    predictions = []

    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("Submit"):
        st.subheader('Result---')
        st.markdown("-------------------")

        # Make predictions
        result = predict_heart_disease(input_data)
        
        # Display the results
        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart Disease Detected")
            else:
                st.write("Heart Disease Detected")
            st.markdown("-----------------------------")

with tab2:
    st.subheader("Bulk Prediction from CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        bulk_data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(bulk_data.head())

        try:
            predictions = []
            for modelname in modelnames:
                model = pickle.load(open(modelname, 'rb'))
                pred = model.predict(bulk_data)
                predictions.append(pred)

            result_df = bulk_data.copy()
            for i in range(len(algonames)):
                result_df[algonames[i]] = ["No Disease" if p == 0 else "Disease" for p in predictions[i]]

            st.subheader("🧾 Prediction Results")
            st.dataframe(result_df)

            # Downloadable CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv, file_name="predicted_results.csv", mime='text/csv')
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Make sure your CSV column names match the expected features used during model training.")

with tab3:
    st.subheader("📊 Model Information")
    st.markdown("""
    This app uses four different machine learning models to predict the likelihood of heart disease:

    - **Decision Tree**: Splits data into branches for easier decision-making.
    - **Logistic Regression**: Calculates probabilities and classifies data.
    - **Random Forest**: A combination of multiple decision trees for more accurate results.
    - **Support Vector Machine (SVM)**: Tries to draw the best boundary between categories.

    Each model makes a separate prediction — giving you a comparative view.
    """)

