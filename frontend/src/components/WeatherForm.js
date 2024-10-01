// WeatherForm.js
import React, { useState } from "react";
import axios from "axios";
import "./WeatherForm.css"; // Import the CSS file

const WeatherForm = () => {
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [windSpeed, setWindSpeed] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/predict", {
        temperature: temperature,
        humidity: humidity,
        wind_speed: windSpeed,
      });
      setPrediction(response.data.weather_event);
    } catch (error) {
      console.error("Error fetching the prediction", error);
    }
  };

  return (
    <div className="form-container">
      <h1>Weather Event Predictor</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Temperature (°C):
          <input
            type="number"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
            required
          />
        </label>
        <label>
          Humidity (%):
          <input
            type="number"
            value={humidity}
            onChange={(e) => setHumidity(e.target.value)}
            required
          />
        </label>
        <label>
          Wind Speed (m/s):
          <input
            type="number"
            value={windSpeed}
            onChange={(e) => setWindSpeed(e.target.value)}
            required
          />
        </label>
        <button type="submit">Submit</button>
      </form>
      {prediction && <div className="prediction-popup">{prediction}</div>}
    </div>
  );
};

export default WeatherForm;
