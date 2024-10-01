// src/components/ResultPopup.js

import React from "react";
import "./ResultPopup.css"; // Import CSS for styling

const ResultPopup = ({ weatherEvent, closePopup }) => {
  return (
    <div className="popup-overlay">
      <div className="popup">
        <h2>Prediction Result</h2>
        <p>
          The predicted weather event is: <strong>{weatherEvent}</strong>
        </p>
        <button onClick={closePopup}>Close</button>
      </div>
    </div>
  );
};

export default ResultPopup;
