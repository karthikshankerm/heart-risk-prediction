import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image

# Page configuration
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")

# Background styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://source.unsplash.com/1920x1080/?heart,healthcare");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.big-font {
    font-size: 36px !important;
    font-weight: 700;
    color: #FF4B4B;
    text-shadow: 1px 1px #000;
    margin-top: 20px;
}

.card {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    font-weight: bold;
    width: 100%;
}

input, select {
    background-color: #222 !important;
    color: #fff !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# Load image using absolute path
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "heart.jpg")
image = Image.open(image_path)
st.image(image, width=1920)

# Load model and encoders
project_root = os.path.abspath(os.path.join(current_dir, ".."))
models_path = os.path.join(project_root, "models")

model = pickle.load(open(os.path.join(models_path, 'xgb_best_model.sav'), 'rb'))
scaler = pickle.load(open(os.path.join(models_path, 'scaler.sav'), 'rb'))
had_angina_label = pickle.load(open(os.path.join(models_path, 'had_angina_label.sav'), 'rb'))
had_arthritis_label = pickle.load(open(os.path.join(models_path, 'had_arthritis_label.sav'), 'rb'))
sex_label = pickle.load(open(os.path.join(models_path, 'sex_label.sav'), 'rb'))
age_category_ohe = pickle.load(open(os.path.join(models_path, 'age_category_ohe.sav'), 'rb'))
had_diabetes_ohe = pickle.load(open(os.path.join(models_path, 'had_diabetes_ohe.sav'), 'rb'))

def main():
    st.markdown("<h1 class='big-font'> Heart Attack Risk Predictor</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.title("💡 About")
        st.write("This app predicts the **risk of heart attack** using key health indicators.")
        st.markdown("""
        **Model uses:**
        - BMI, Sleep, Mental & Physical health
        - Age, Sex, Diabetes status
        - Arthritis and Angina history
        """)
        st.info("Try changing inputs to see how risk level varies!")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.form(key="risk_form"):
            col1, col2 = st.columns(2)

            with col1:
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
                sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, format="%.1f")
                physical_health_days = st.number_input("Physical Health Bad Days (last 30 days)", 0, 30, value=5)
                mental_health_days = st.number_input("Mental Health Bad Days (last 30 days)", 0, 30, value=5)

            with col2:
                had_angina = st.selectbox("Had Angina?", ["No", "Yes"])
                had_arthritis = st.selectbox("Had Arthritis?", ["No", "Yes"])
                age_category = st.selectbox("Age Category", ["Young", "Middle-Aged", "Old"])
                sex = st.selectbox("Sex", ["Male", "Female"])
                had_diabetes = st.selectbox("Had Diabetes?", ["No", "Yes", "Borderline"])

            submit = st.form_submit_button(" Predict Risk")

        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            try:
                # Encoding
                had_angina_encoded = had_angina_label.transform([had_angina])[0]
                had_arthritis_encoded = had_arthritis_label.transform([had_arthritis])[0]
                sex_encoded = sex_label.transform([sex])[0]
                age_encoded = age_category_ohe.transform([[age_category]])[0]
                had_diabetes_encoded = had_diabetes_ohe.transform([[had_diabetes]])[0]

                # Extract one-hot parts
                age_old = age_encoded[2] if len(age_encoded) == 3 else 0
                age_young = age_encoded[0] if len(age_encoded) == 3 else 0
                had_diabetes_no = had_diabetes_encoded[0] if len(had_diabetes_encoded) == 3 else 0

                numeric = np.array([bmi, sleep_hours, physical_health_days, mental_health_days])

                final_input = np.array([
                    numeric[0], had_angina_encoded, numeric[1],
                    numeric[2], age_old, numeric[3],
                    age_young, had_arthritis_encoded,
                    sex_encoded, had_diabetes_no
                ]).reshape(1, -1)

                scaled = scaler.transform(final_input)
                prediction = model.predict(scaled)[0]
                result = "High" if prediction == 1 else "Low"

                st.success(f"✅ Predicted Heart Attack Risk: **{result}**")

            except Exception as e:
                st.error(f" Prediction Error: {e}")

# Run the app
if __name__ == "__main__":
    main()
