from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('best_model_pipeline.pkl')

@app.route('/')
def home():
    return "🎉 ML Model API is running! Use POST /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df).tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
