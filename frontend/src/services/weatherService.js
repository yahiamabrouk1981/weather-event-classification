import axios from "axios";

const API_URL = "http://localhost:3001";

const predictWeather = async (temperature, humidity, wind_speed) => {
  const response = await axios.post(`${API_URL}/predict`, {
    temperature,
    humidity,
    wind_speed,
  });
  return response.data;
};

export default {
  predictWeather,
};
