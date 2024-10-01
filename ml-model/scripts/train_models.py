import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Importing joblib for saving models

# Function to map weather events
def map_weather_event(row):
    precip_type = str(row['Precip Type']).lower()  # Handle possible float issue by converting to string
    summary = row['Summary'].lower()

    if 'clear' in summary or 'sun' in summary:
        return 'Clear'
    elif 'cloud' in summary or 'overcast' in summary:
        return 'Cloudy'
    elif 'rain' in precip_type:
        return 'Rain'
    elif 'snow' in precip_type:
        return 'Snow'
    else:
        return 'Other'

# Load the dataset
df = pd.read_csv('./../data/weather.csv')

# Create new 'weather_event' column
df['weather_event'] = df.apply(map_weather_event, axis=1)

# Upsampling the minority class
df_majority = df[df.weather_event == 'Cloudy']
df_minority_clear = df[df.weather_event == 'Clear']
df_minority_rain = df[df.weather_event == 'Rain']
df_minority_snow = df[df.weather_event == 'Snow']

df_minority_clear_upsampled = resample(df_minority_clear, replace=True, n_samples=len(df_majority), random_state=42)
df_minority_rain_upsampled = resample(df_minority_rain, replace=True, n_samples=len(df_majority), random_state=42)
df_minority_snow_upsampled = resample(df_minority_snow, replace=True, n_samples=len(df_majority), random_state=42)

# Combine all classes into a single DataFrame
df_balanced = pd.concat([df_majority, df_minority_clear_upsampled, df_minority_rain_upsampled, df_minority_snow_upsampled])

# Feature engineering: Selecting relevant features and creating new ones (e.g., Apparent Temp - Temp difference)
df_balanced['Temp Difference (C)'] = df_balanced['Apparent Temperature (C)'] - df_balanced['Temperature (C)']
X = df_balanced[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Temp Difference (C)']]
y = df_balanced['weather_event']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# KNN Model
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate Models with Cross-Validation
def cross_val(model, X_train, y_train, model_name):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Logistic Regression Cross-Validation
cross_val(log_reg, X_train, y_train, "Logistic Regression")

# Random Forest Cross-Validation
cross_val(best_rf, X_train, y_train, "Random Forest")

# KNN Cross-Validation
cross_val(knn, X_train_scaled, y_train, "KNN")

# Predict using the models
y_pred_rf = best_rf.predict(X_test)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_log_reg = log_reg.predict(X_test)

# Evaluation function
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Logistic Regression evaluation
evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")

# Random Forest evaluation
evaluate_model(y_test, y_pred_rf, "Random Forest")

# KNN evaluation
evaluate_model(y_test, y_pred_knn, "KNN")

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the models
joblib.dump(best_rf, 'random_forest_model.joblib')
joblib.dump(knn, 'knn_model.joblib')
joblib.dump(log_reg, 'logistic_regression_model.joblib')

print("Models saved successfully!")
