from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
from io import BytesIO
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import logging

# Define stopwords
STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models and preprocessors once at startup
with open(r"model/model_xgb.pkl", "rb") as model_file:
    predictor = pickle.load(model_file)

with open(r"model/MinMaxScaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(r"model/CountVectorizer.pkl", "rb") as cv_file:
    cv = pickle.load(cv_file)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif request.is_json and "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "Invalid input. Please provide a file or a JSON payload with 'text'."}), 400

    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500

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

    return predictions_csv, graph

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

if __name__ == "__main__":
    app.run(port=5000, debug=True)
