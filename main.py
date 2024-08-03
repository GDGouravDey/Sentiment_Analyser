import base64
import logging
import re
from io import BytesIO

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize nltk resources
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models and preprocessors once at startup
with open(r"model/model_xgb.pkl", "rb") as model_file:
    predictor = pickle.load(model_file)

with open(r"model/MinMaxScaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(r"model/CountVectorizer.pkl", "rb") as cv_file:
    cv = pickle.load(cv_file)


def single_prediction(predictor, scaler, cv, text_input):
    """Predict sentiment for a single text input."""
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    """Predict sentiment for bulk data in CSV."""
    corpus = []
    stemmer = PorterStemmer()
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return data, predictions_csv, graph


def get_distribution_graph(data):
    """Generate a pie chart of sentiment distribution."""
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    """Map numerical sentiment prediction to text label."""
    return "Positive" if x == 1 else "Negative"


def run_streamlit_app():
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
                # Bulk prediction from CSV file
                data = pd.read_csv(uploaded_file)
                prediction_data, response_bytes, graph = bulk_prediction(
                    predictor, scaler, cv, data
                )

                # Display the DataFrame
                st.write("Prediction Results:")
                st.dataframe(prediction_data)

                # Show the distribution graph
                st.image(graph.getvalue(), format="PNG")

                # Download button for the predictions
                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    key="result_download_button",
                )

            elif user_input:
                # Single string prediction
                predicted_sentiment = single_prediction(
                    predictor, scaler, cv, user_input
                )
                st.write(f"Predicted sentiment: {predicted_sentiment}")

            else:
                st.warning("Please enter text or upload a CSV file for prediction.")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_streamlit_app()
