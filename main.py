import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Set the prediction endpoint URL
prediction_endpoint = "http://127.0.0.1:5000/predict"

# Title of the Streamlit app
st.title("Text Sentiment Predictor")

# File uploader for bulk prediction
uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

# Text input for single sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Button to trigger prediction
if st.button("Predict"):
    try:
        if uploaded_file is not None:
            # Prepare file for bulk prediction
            file = {"file": uploaded_file}
            response = requests.post(prediction_endpoint, files=file)
            
            if response.status_code == 200:
                # Load CSV response as a DataFrame
                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)

                # Display the DataFrame
                st.write("Prediction Results:")
                st.dataframe(response_df)

                # Download button for the predictions
                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    key="result_download_button",
                )
            else:
                st.error("Failed to get predictions for the uploaded file.")
                st.error(f"Error: {response.text}")

        elif user_input:
            # Prepare JSON payload for single prediction
            response = requests.post(
                prediction_endpoint,
                json={"text": user_input}  # Use JSON format
            )

            if response.status_code == 200:
                response_data = response.json()
                st.write(f"Predicted sentiment: {response_data['prediction']}")
            else:
                st.error("Failed to get prediction for the entered text.")
                st.error(f"Error: {response.text}")

        else:
            st.warning("Please enter text or upload a CSV file for prediction.")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the prediction API: {e}")
