from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

# Load the trained Random Forest model
model = joblib.load('./models/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = data['temperature']
    humidity = data['humidity']
    wind_speed = data['wind_speed']
    
    # Prepare the input for prediction
    input_features = np.array([[temperature, humidity, wind_speed]])
    
    # Predict the weather event
    prediction = model.predict(input_features)
    return jsonify({'weather_event': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
