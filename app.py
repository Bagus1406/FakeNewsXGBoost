import streamlit as st
import pickle
import numpy as np
import shap

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
    return probs

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
                probs = predict_with_prob(user_input)
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
                X_input = vectorizer.transform([user_input])
                # Buat explainer hanya untuk input ini
                explainer = shap.Explainer(model, X_input)
                shap_values = explainer([X_input])

                # Display SHAP bar chart
                fig, ax = plt.subplots()
                shap.plots.bar(shap_values[0], max_display=10, show=False)
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
