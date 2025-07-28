import joblib
import numpy as np

def load_models():
    classification_model = joblib.load('models/poverty_classification_model_v1.0_20250728_143203.pkl')
    regression_model = joblib.load('models/poverty_regression_model_v1.0_20250728_143203.pkl')
    scaler = joblib.load('models/scaler_v1.0_20250728_143203.pkl')
    return classification_model, regression_model, scaler

def predict_poverty_level(classification_model, input_data):
    prediction = classification_model.predict(input_data)
    return prediction[0]

def predict_mpi_hcr(regression_model, scaler, input_data):
    scaled_data = scaler.transform(input_data)
    prediction = regression_model.predict(scaled_data)
    return prediction[0]

def preprocess_input(data):
    # Extract features in the order expected by your model
    feature_order = [
        "households", "totalPopulation", "totalMales", "totalFemales",
        "literatePopulation", "literateMales", "literateFemales",
        "illiteratePopulation", "maleIlliterates", "femaleIlliterates",
        "totalWorkingPopulation", "totalWorkingMales", "totalWorkingFemales",
        "unemployedPopulation", "unemployedMales", "unemployedFemales",
        "st", "sc", "hindus", "muslims", "sikhs", "buddhists", "jains",
        "othersReligions", "religionNotStated", "households1",
        "ruralHouseholds", "urbanHouseholds", "householdsWithInternet"
    ]
    # Convert values to float (or int as needed)
    features = [float(data.get(feat, 0)) for feat in feature_order]
    return features