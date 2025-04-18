import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model_pipeline.pkl')

# Page configuration
st.set_page_config(page_title="Student Score Predictor", layout="centered")

# Main title
st.markdown("<h1 style='text-align: center;'>📊 Student Math Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("🧾 Enter Student Info")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "associate's degree",
    "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, value=72)
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, value=74)

# Predict button
if st.sidebar.button("📌 Predict Math Score"):
    input_data = {
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_level_of_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    }

    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]

    # Display prediction as a styled clipboard/card
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 6px solid #00cc99; margin-top: 20px;'>
            <h3 style='color: #00cc99;'>📈 Predicted Math Score</h3>
            <p style='font-size: 32px; font-weight: bold;'>🎯 {:.2f}</p>
        </div>
    """.format(prediction), unsafe_allow_html=True)

else:
    st.markdown("👈 Fill out the form on the left and click **Predict Math Score** to see results.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
