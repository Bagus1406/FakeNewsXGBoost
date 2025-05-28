import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load model, vectorizer, dan label encoder
@st.cache_resource
def load_components():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, vectorizer, le

model, vectorizer, le = load_components()

# Prediksi dengan probabilitas
def predict_with_prob(text):
    X_input = vectorizer.transform([text])
    probs = model.predict_proba(X_input)[0]
    return probs, X_input

# Streamlit UI
def main():
    st.markdown(
        """<div style="background-color:#2c3e50;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center">Fake News Detector</h1> 
            <h4 style="color:white;text-align:center">Built with XGBoost & Streamlit</h4> 
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """### Welcome!
This app uses a **TF-IDF + XGBoost** model to classify news content as **FAKE** or **REAL**.  
#### Model Info  
Trained on 54k+ preprocessed English-language news samples.
""",
        unsafe_allow_html=True,
    )

    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Fake News Prediction")
        user_input = st.text_area("Enter a news headline or article (English only):")

        if st.button("Detect"):
            if user_input.strip() == "":
                st.warning("Text cannot be empty.")
            else:
                probs, X_input = predict_with_prob(user_input)
                pred_index = int(np.argmax(probs))
                pred_label = le.inverse_transform([pred_index])[0]
                confidence = round(probs[pred_index] * 100, 2)

                if pred_label == "REAL":
                    st.success(f"Prediction: **{pred_label}** ({confidence}% confident)")
                else:
                    st.error(f"Prediction: **{pred_label}** ({confidence}% confident)")

                st.markdown("#### Confidence per class")
                st.bar_chart({
                    "Confidence": {
                        le.inverse_transform([0])[0]: probs[0],
                        le.inverse_transform([1])[0]: probs[1]
                    }
                })

                # SHAP explanation
                st.subheader("Why this prediction?")
                explainer = shap.TreeExplainer(model.get_booster())
                X_input_array = X_input.toarray()
                shap_values = explainer.shap_values(X_input_array)

                # Ambil top N fitur
                N = 10
                feature_names = vectorizer.get_feature_names_out()
                shap_dict = dict(zip(vectorizer.get_feature_names_out(), shap_values[0]))
                shap_df = pd.DataFrame.from_dict(shap_dict, orient='index', columns=['shap_value'])
                shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
                shap_df = shap_df.head(N)

                # Plot pakai matplotlib (seperti confidence bar)
                # Plot pakai matplotlib (match dark mode)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#1f77b4' if val > 0 else '#E74C3C' for val in shap_df.shap_value]

                ax.barh(shap_df.index[::-1], shap_df.shap_value[::-1], color=colors[::-1])
                ax.set_xlabel("SHAP Value", color='white')
                ax.set_title("Top Feature Contributions", color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                fig.patch.set_facecolor('#2c3e50')     # Sesuai background
                ax.set_facecolor('#2c3e50')            # Area plot

                st.pyplot(fig)


    # Credit
    st.markdown("""---""")
    st.markdown(
        """
        <div style="text-align: center; font-size: 14px;">
            <p><strong>Created by Team Sigma Male</strong></p>
            <ul style="list-style-type: none; padding: 0;">
                <li>1. Hafidz Akbar Faridzi R.</li>
                <li>2. Muhammad Bagus Kurniawan</li>
                <li>3. Nurul Alpi Najam</li>
                <li>4. Ryan Rasyid Azizi</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
