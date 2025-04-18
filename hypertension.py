import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open('ML_Models/hypertension.pkl', 'rb'))

# Initialize scaler with typical ranges for medical features
# These ranges are based on typical medical values for each feature
scaler = StandardScaler()
scaler.mean_ = np.array([1.5, 130, 250, 150, 1.0, 1.0, 0.5])  # Typical means
scaler.scale_ = np.array([1.0, 20, 50, 20, 1.0, 0.5, 0.5])    # Typical standard deviations

# Define expected feature names and order
FEATURE_NAMES = ['cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

class Hypertension:
    def __init__(self, data):
        """
        Initialize with input data dictionary.
        
        Parameters:
        data (dict): Dictionary containing the following keys:
            - cp (int): Chest pain type
            - trestbps (int): Resting blood pressure
            - chol (int): Cholesterol
            - thalach (int): Maximum heart rate achieved
            - oldpeak (float): ST depression induced by exercise relative to rest
            - slope (int): Slope of the peak exercise ST segment
            - ca (int): Number of major vessels colored by fluoroscopy
        """
        self.result = "Error making prediction"
        try:
            # Check if all required keys are present
            missing_keys = [key for key in FEATURE_NAMES if key not in data]
            if missing_keys:
                self.result = f"Missing required keys: {', '.join(missing_keys)}"
                return

            # Extract features in correct order
            features = np.array([[
                float(data['cp']),
                float(data['trestbps']),
                float(data['chol']),
                float(data['thalach']),
                float(data['oldpeak']),
                float(data['slope']),
                float(data['ca'])
            ]])
            
            # Print input features for debugging
            print("Input features:", features)
            
            # Scale the features using predefined parameters
            scaled_features = scaler.transform(features)
            print("Scaled features:", scaled_features)
            
            # Make prediction
            prediction = int(model.predict(scaled_features)[0])
            print("Raw prediction:", prediction)
            
            # Get prediction probabilities if available
            try:
                proba = model.predict_proba(scaled_features)[0]
                print("Prediction probabilities:", proba)
            except:
                print("Model does not support probability predictions")
            
            # Fix: Reverse the prediction interpretation to match medical logic
            # 0 = Hypertension, 1 = No Hypertension in the model
            self.result = "No Hypertension" if prediction == 1 else "Hypertension"
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            self.result = f"Error making prediction: {str(e)}"
    
    def __str__(self):
        """Return the prediction result as a string."""
        return self.result

    def to_dict(self):
        """Return the prediction result as a dictionary for JSON serialization."""
        return {"result": self.result}