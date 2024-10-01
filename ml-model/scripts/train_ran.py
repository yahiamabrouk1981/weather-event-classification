import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Function to map weather events based on 'Precip Type'
def map_weather_event(row):
    precip_type = str(row['Precip Type']).lower()  # Ensure precip_type is a string
    if precip_type == 'rain':
        return 'Rain'
    elif precip_type == 'snow':
        return 'Snow'
    elif precip_type in ['clear', 'partly cloudy', 'mostly cloudy']:
        return 'Clear' if 'clear' in precip_type else 'Cloudy'
    else:
        return 'Unknown'  # For any unexpected types

# Load dataset
df = pd.read_csv('./../data/weather.csv')

# Add a new column for weather events
df['weather_event'] = df.apply(map_weather_event, axis=1)

# Select features and target variable
X = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
y = df['weather_event']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f'Random Forest Cross-Validation Accuracy: {cv_scores.mean():.4f}')

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Random Forest Evaluation:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Save the trained model
joblib.dump(rf_model, './../models/random_forest_model.pkl')
print(df['weather_event'])
print(df['weather_event'].value_counts())
print("Model saved as 'random_forest_model.pkl'")
