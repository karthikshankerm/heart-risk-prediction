# 💖 Heart Attack Risk Prediction App  
🔗 Live Demo: [heart-risk-prediction.streamlit.app](https://heart-risk-prediction-norgnh8ony7nyet2em3hgx.streamlit.app)

An interactive machine learning web app built with **Streamlit** to predict a person’s risk of having a heart attack based on key health indicators.

---

## 🔍 Features

- Predicts **Heart Attack Risk** using a trained **XGBoost Classifier**
- Clean and responsive Streamlit UI
- Encodes categorical and scaled inputs automatically
- Real-time predictions using `.sav` model files

---

## 💻 Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas, NumPy
- Pillow (for image handling)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/karthikshankerm/heart-risk-prediction.git
cd heart-risk-prediction
pip install -r requirements.txt
streamlit run app/app.py
