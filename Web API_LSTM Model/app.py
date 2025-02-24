set TF_ENABLE_ONEDNN_OPTS=0
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify

# Load the trained model
model = load_model("imdb_lstm_model.h5")

# Initialize Flask app
app = Flask(__name__)

# Load IMDB word index for text processing
word_index = tf.keras.datasets.imdb.get_word_index()
vocab_size = 10000
max_len = 200

def preprocess_text(text):
    """Convert input text to a padded sequence."""
    words = text.lower().split()
    sequence = [[word_index.get(word, 2) for word in words]]  # 2 is for <UNK> words
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    return padded_sequence

@app.route("/", methods=["GET"])
def home():
    return "LSTM Sentiment Analysis API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.json
        review = data.get("review", "")

        # Preprocess and predict
        processed_review = preprocess_text(review)
        prediction = model.predict(processed_review)[0][0]
        sentiment = "Negative" if prediction >= 0.5 else "positive"

        # Return JSON response
        return jsonify({"sentiment": sentiment, "confidence": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
