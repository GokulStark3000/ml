import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data
data = pd.read_csv("hypertension_data.csv")

# Prepare features
feature_names = ['cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']
X = data[feature_names]
y = data['target']

# Create and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the model
model = SVC(probability=True)
model.fit(X_scaled, y)

# Save both model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}

with open('hypertension.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and scaler saved successfully!") 