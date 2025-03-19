import joblib

# Load the scaler
scaler = joblib.load("scaler.pkl")
print(type(scaler))
