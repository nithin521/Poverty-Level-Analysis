# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# from flask_cors import CORS
# import os
# import json
# import shap

# app = Flask(__name__)
# CORS(app)
# feature_metadata = {
#     'literacy_rate': 'Higher literacy reduces poverty',
#     'female_literacy_rate': 'Educated women improve family and community well-being',
#     'male_literacy_rate': 'Educated men reduce poverty burden',
#     'unemployment_rate': 'Higher unemployment increases poverty',
#     'female_unemployment_rate': 'Female joblessness impacts household income',
#     'male_unemployment_rate': 'Male joblessness contributes to income poverty',
#     'working_participation_rate': 'More employment reduces poverty',
#     'female_working_rate': 'Women’s employment helps poverty alleviation',
#     'internet_penetration': 'Digital access reduces poverty',
#     'urban_ratio': 'Urban areas tend to have lower poverty',
#     'rural_ratio': 'Rural areas tend to have higher poverty',
#     'population_density': 'High density may relate to urban advantages',
#     'st_sc_ratio': 'Marginalized communities often face higher poverty',
#     'gender_ratio': 'Balanced gender ratio is key to equity',
#     'household_size': 'Large household size can strain resources',
#     'Hindus_ratio': 'Cultural demographic factor',
#     'Muslims_ratio': 'Cultural demographic factor',
#     'Sikhs_ratio': 'Cultural demographic factor',
#     'Buddhists_ratio': 'Cultural demographic factor',
#     'Jains_ratio': 'Cultural demographic factor',
#     'Others_Religions_ratio': 'Cultural demographic factor'
# }


# class PovertyPredictionAPI:
#     def __init__(self):
#         self.classification_model = None
#         self.regression_model = None
#         self.scaler = None
#         self.feature_names = []
#         self.model_loaded = False
        
#     def load_models(self):
#         """Load the trained models and scaler"""
#         try:
#             # Load models
#             self.classification_model = joblib.load('models/poverty_classification_model_v1.0_20250728_143203.pkl')
#             self.regression_model = joblib.load('models/poverty_regression_model_v1.0_20250728_143203.pkl')
            
#             # Try to load the scaler
#             try:
#                 self.scaler = joblib.load('models/scaler_v1.0_20250728_143203.pkl')
#                 print("✓ Scaler loaded successfully")
#             except Exception as scaler_error:
#                 print(f"⚠ Warning: Could not load scaler: {scaler_error}")
#                 print("Will create a new scaler or use feature scaling manually")
#                 self.scaler = None
            
#             # Try to load metadata to get feature names
#             try:
#                 with open('models/model_metadata_v1.0_20250728_143203.json', 'r') as f:
#                     metadata = json.load(f)
#                     self.feature_names = metadata.get('regression', {}).get('feature_names', [])
#                     print(f"✓ Loaded {len(self.feature_names)} feature names from metadata")
#             except Exception as meta_error:
#                 print(f"⚠ Warning: Could not load metadata: {meta_error}")
#                 # Fallback feature names based on your code
#                 self.feature_names = [
#                     'literacy_rate', 'female_literacy_rate', 'male_literacy_rate',
#                     'unemployment_rate', 'female_unemployment_rate', 'male_unemployment_rate',
#                     'working_participation_rate', 'female_working_rate',
#                     'internet_penetration', 'urban_ratio', 'rural_ratio',
#                     'population_density', 'st_sc_ratio', 'gender_ratio', 'household_size',
#                     'Hindus_ratio', 'Muslims_ratio', 'Sikhs_ratio', 'Buddhists_ratio',
#                     'Jains_ratio', 'Others_Religions_ratio'
#                 ]
            
#             self.model_loaded = True
#             print("✓ Models loaded successfully")
#             return True
            
#         except Exception as e:
#             print(f"✗ Error loading models: {str(e)}")
#             self.model_loaded = False
#             return False
    
#     def preprocess_input(self, data):
#         """Convert input data to engineered features"""
#         try:
#             # Extract values from input data
#             area = float(data.get('area', 0))
#             households = float(data.get('households', 0))
#             total_population = float(data.get('totalPopulation', 0))
#             total_males = float(data.get('totalMales', 0))
#             total_females = float(data.get('totalFemales', 0))
#             literate_population = float(data.get('literatePopulation', 0))
#             literate_males = float(data.get('literateMales', 0))
#             literate_females = float(data.get('literateFemales', 0))
#             total_working_population = float(data.get('totalWorkingPopulation', 0))
#             total_working_females = float(data.get('totalWorkingFemales', 0))
#             unemployed_population = float(data.get('unemployedPopulation', 0))
#             unemployed_males = float(data.get('unemployedMales', 0))
#             unemployed_females = float(data.get('unemployedFemales', 0))
#             st = float(data.get('st', 0))
#             sc = float(data.get('sc', 0))
#             hindus = float(data.get('hindus', 0))
#             muslims = float(data.get('muslims', 0))
#             sikhs = float(data.get('sikhs', 0))
#             buddhists = float(data.get('buddhists', 0))
#             jains = float(data.get('jains', 0))
#             others_religions = float(data.get('othersReligions', 0))
#             rural_households = float(data.get('ruralHouseholds', 0))
#             urban_households = float(data.get('urbanHouseholds', 0))
#             households_with_internet = float(data.get('householdsWithInternet', 0))
            
#             # Calculate engineered features (same as in your training code)
#             features = {}
            
#             # Avoid division by zero
#             if total_population > 0:
#                 features['literacy_rate'] = (literate_population / total_population) * 100
#                 features['unemployment_rate'] = (unemployed_population / total_population) * 100
#                 features['working_participation_rate'] = (total_working_population / total_population) * 100
#                 features['st_sc_ratio'] = ((st + sc) / total_population) * 100
#                 features['Hindus_ratio'] = (hindus / total_population) * 100
#                 features['Muslims_ratio'] = (muslims / total_population) * 100
#                 features['Sikhs_ratio'] = (sikhs / total_population) * 100
#                 features['Buddhists_ratio'] = (buddhists / total_population) * 100
#                 features['Jains_ratio'] = (jains / total_population) * 100
#                 features['Others_Religions_ratio'] = (others_religions / total_population) * 100
#             else:
#                 features.update({
#                     'literacy_rate': 0, 'unemployment_rate': 0, 'working_participation_rate': 0,
#                     'st_sc_ratio': 0, 'Hindus_ratio': 0, 'Muslims_ratio': 0, 'Sikhs_ratio': 0,
#                     'Buddhists_ratio': 0, 'Jains_ratio': 0, 'Others_Religions_ratio': 0
#                 })
            
#             if total_females > 0:
#                 features['female_literacy_rate'] = (literate_females / total_females) * 100
#                 features['female_unemployment_rate'] = (unemployed_females / total_females) * 100
#                 features['female_working_rate'] = (total_working_females / total_females) * 100
#             else:
#                 features.update({
#                     'female_literacy_rate': 0, 'female_unemployment_rate': 0, 'female_working_rate': 0
#                 })
            
#             if total_males > 0:
#                 features['male_literacy_rate'] = (literate_males / total_males) * 100
#                 features['male_unemployment_rate'] = (unemployed_males / total_males) * 100
#                 features['gender_ratio'] = (total_females / total_males) * 1000
#             else:
#                 features.update({
#                     'male_literacy_rate': 0, 'male_unemployment_rate': 0, 'gender_ratio': 0
#                 })
            
#             if households > 0:
#                 features['internet_penetration'] = (households_with_internet / households) * 100
#                 features['urban_ratio'] = (urban_households / households) * 100
#                 features['rural_ratio'] = (rural_households / households) * 100
#                 features['household_size'] = total_population / households
#             else:
#                 features.update({
#                     'internet_penetration': 0, 'urban_ratio': 0, 'rural_ratio': 0, 'household_size': 0
#                 })
            
#             if area > 0:
#                 features['population_density'] = total_population / area
#             else:
#                 features['population_density'] = 0
            
#             # Convert to array in the correct order
#             feature_array = []
#             for feature_name in self.feature_names:
#                 feature_array.append(features.get(feature_name, 0))
            
#             return np.array(feature_array)
            
#         except Exception as e:
#             print(f"Error in preprocessing: {str(e)}")
#             raise e
        

#     def explain_prediction(self, scaled_features):
#         try:
#             explainer = shap.TreeExplainer(self.classification_model)
#             shap_values = explainer.shap_values(scaled_features)

#             # Handle binary vs multiclass
#             if isinstance(shap_values, list) and len(shap_values) == 2:
#                 values = shap_values[1][0]  # Class 1 (High)
#             else:
#                 values = shap_values[0]  # If only one output (regressor or binary fallback)
#                 if len(values.shape) > 1:
#                     values = values[0]

#             explanation = []
#             for i, feat_name in enumerate(self.feature_names):
#                 val = values[i]
#                 explanation.append({
#                     'feature': feat_name,
#                     'importance': round(abs(val), 4),
#                     'impact': 'positive' if val > 0 else 'negative',
#                     'description': feature_metadata.get(feat_name, '')
#                 })

#             explanation = sorted(explanation, key=lambda x: x['importance'], reverse=True)
#             return explanation

#         except Exception as e:
#             print(f"⚠ SHAP explanation failed: {e}")
#             return []

    
#     def predict(self, data):
#         """Make predictions using the loaded models"""
#         if not self.model_loaded:
#             raise Exception("Models not loaded properly")
        
#         try:
#             # Preprocess input
#             input_features = self.preprocess_input(data)
            
#             # Reshape for prediction
#             input_features = input_features.reshape(1, -1)
            
#             # Scale features if scaler is available
#             if self.scaler is not None:
#                 try:
#                     scaled_features = self.scaler.transform(input_features)
#                     print("✓ Features scaled using loaded scaler")
#                 except Exception as scale_error:
#                     print(f"⚠ Warning: Scaler transform failed: {scale_error}")
#                     print("Using unscaled features")
#                     scaled_features = input_features
#             else:
#                 print("⚠ No scaler available, using unscaled features")
#                 scaled_features = input_features
            
#             # Make predictions
#             mpi_hcr = self.regression_model.predict(scaled_features)[0]
#             poverty_level = self.classification_model.predict(scaled_features)[0]
            
#             # Get prediction probabilities for classification
#             try:
#                 poverty_prob = self.classification_model.predict_proba(scaled_features)[0]
#                 prob_low = poverty_prob[0]
#                 prob_high = poverty_prob[1]
#             except:
#                 prob_low = 0.5
#                 prob_high = 0.5
            
#             explanation = self.explain_prediction(scaled_features)
#             return {
#                 'mpiHcr': float(mpi_hcr),
#                 'povertyLevel': 'High' if poverty_level == 1 else 'Low',
#                 'confidence': {
#                     'low_risk_probability': float(prob_low),
#                     'high_risk_probability': float(prob_high)
#                 },
#                 'explanation': explanation
#             }

            
#         except Exception as e:
#             print(f"Error in prediction: {str(e)}")
#             raise e

# # Initialize the API
# api = PovertyPredictionAPI()

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data
#         data = request.json
#         print(f"Received data: {data}")
        
#         # Validate input
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
        
#         # Make prediction
#         result = api.predict(data)
#         print(f"Prediction result: {result}")
        
#         return jsonify(result)
        
#     except Exception as e:
#         error_message = f"Prediction failed: {str(e)}"
#         print(error_message)
#         return jsonify({'error': error_message}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': api.model_loaded,
#         'scaler_available': api.scaler is not None,
#         'feature_count': len(api.feature_names)
#     })

# @app.route('/features', methods=['GET'])
# def get_features():
#     """Get the list of features expected by the model"""
#     return jsonify({
#         'features': api.feature_names,
#         'count': len(api.feature_names)
#     })


# if __name__ == '__main__':
#     # Load models at startup
#     print("Starting Poverty Prediction API...")
#     api.load_models()
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import json
import shap

app = Flask(__name__)
CORS(app)
feature_metadata = {
    'literacy_rate': 'Higher literacy reduces poverty',
    'female_literacy_rate': 'Educated women improve family and community well-being',
    'male_literacy_rate': 'Educated men reduce poverty burden',
    'unemployment_rate': 'Higher unemployment increases poverty',
    'female_unemployment_rate': 'Female joblessness impacts household income',
    'male_unemployment_rate': 'Male joblessness contributes to income poverty',
    'working_participation_rate': 'More employment reduces poverty',
    'female_working_rate': 'Women’s employment helps poverty alleviation',
    'internet_penetration': 'Digital access reduces poverty',
    'urban_ratio': 'Urban areas tend to have lower poverty',
    'rural_ratio': 'Rural areas tend to have higher poverty',
    'population_density': 'High density may relate to urban advantages',
    'gender_ratio': 'Balanced gender ratio is key to equity',
    'household_size': 'Large household size can strain resources',

}


class PovertyPredictionAPI:
    def __init__(self):
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.feature_names = []
        self.model_loaded = False
        
    def load_models(self):
        """Load the trained models and scaler"""
        try:
            # Load models
            self.classification_model = joblib.load('models/poverty_classification_model_v1.0_20250728_143203.pkl')
            self.regression_model = joblib.load('models/poverty_regression_model_v1.0_20250728_143203.pkl')
            
            # Try to load the scaler
            try:
                self.scaler = joblib.load('models/scaler_v1.0_20250728_143203.pkl')
                print("✓ Scaler loaded successfully")
            except Exception as scaler_error:
                print(f"⚠ Warning: Could not load scaler: {scaler_error}")
                print("Will create a new scaler or use feature scaling manually")
                self.scaler = None
            
            # Try to load metadata to get feature names
            try:
                with open('models/model_metadata_v1.0_20250728_143203.json', 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('regression', {}).get('feature_names', [])
                    print(f"✓ Loaded {len(self.feature_names)} feature names from metadata")
            except Exception as meta_error:
                print(f"⚠ Warning: Could not load metadata: {meta_error}")
                # Fallback feature names based on your code
                self.feature_names = [
                    'literacy_rate', 'female_literacy_rate', 'male_literacy_rate',
                    'unemployment_rate', 'female_unemployment_rate', 'male_unemployment_rate',
                    'working_participation_rate', 'female_working_rate',
                    'internet_penetration', 'urban_ratio', 'rural_ratio',
                    'population_density', 'st_sc_ratio', 'gender_ratio', 'household_size',
                    'Hindus_ratio', 'Muslims_ratio', 'Sikhs_ratio', 'Buddhists_ratio',
                    'Jains_ratio', 'Others_Religions_ratio'
                ]
            
            self.model_loaded = True
            print("✓ Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {str(e)}")
            self.model_loaded = False
            return False
    
    def preprocess_input(self, data):
        """Convert input data to engineered features"""
        try:
            # Extract values from input data
            area = float(data.get('area', 0))
            households = float(data.get('households', 0))
            total_population = float(data.get('totalPopulation', 0))
            total_males = float(data.get('totalMales', 0))
            total_females = float(data.get('totalFemales', 0))
            literate_population = float(data.get('literatePopulation', 0))
            literate_males = float(data.get('literateMales', 0))
            literate_females = float(data.get('literateFemales', 0))
            total_working_population = float(data.get('totalWorkingPopulation', 0))
            total_working_females = float(data.get('totalWorkingFemales', 0))
            unemployed_population = float(data.get('unemployedPopulation', 0))
            unemployed_males = float(data.get('unemployedMales', 0))
            unemployed_females = float(data.get('unemployedFemales', 0))
            rural_households = float(data.get('ruralHouseholds', 0))
            urban_households = float(data.get('urbanHouseholds', 0))
            households_with_internet = float(data.get('householdsWithInternet', 0))
            
            # Calculate engineered features (same as in your training code)
            features = {}
            
            # Avoid division by zero
            if total_population > 0:
                features['literacy_rate'] = (literate_population / total_population) * 100
                features['unemployment_rate'] = (unemployed_population / total_population) * 100
                features['working_participation_rate'] = (total_working_population / total_population) * 100
            else:
                features.update({
                    'literacy_rate': 0, 'unemployment_rate': 0, 'working_participation_rate': 0
                })
            
            if total_females > 0:
                features['female_literacy_rate'] = (literate_females / total_females) * 100
                features['female_unemployment_rate'] = (unemployed_females / total_females) * 100
                features['female_working_rate'] = (total_working_females / total_females) * 100
            else:
                features.update({
                    'female_literacy_rate': 0, 'female_unemployment_rate': 0, 'female_working_rate': 0
                })
            
            if total_males > 0:
                features['male_literacy_rate'] = (literate_males / total_males) * 100
                features['male_unemployment_rate'] = (unemployed_males / total_males) * 100
                features['gender_ratio'] = (total_females / total_males) * 1000
            else:
                features.update({
                    'male_literacy_rate': 0, 'male_unemployment_rate': 0, 'gender_ratio': 0
                })
            
            if households > 0:
                features['internet_penetration'] = (households_with_internet / households) * 100
                features['urban_ratio'] = (urban_households / households) * 100
                features['rural_ratio'] = (rural_households / households) * 100
                features['household_size'] = total_population / households
            else:
                features.update({
                    'internet_penetration': 0, 'urban_ratio': 0, 'rural_ratio': 0, 'household_size': 0
                })
            
            if area > 0:
                features['population_density'] = total_population / area
            else:
                features['population_density'] = 0
            
            # Convert to array in the correct order
            feature_array = []
            for feature_name in self.feature_names:
                feature_array.append(features.get(feature_name, 0))
            
            return np.array(feature_array)
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise e
        

    def explain_prediction(self, scaled_features):
        try:
            explainer = shap.TreeExplainer(self.classification_model)
            shap_values = explainer.shap_values(scaled_features)

            # Handle binary vs multiclass
            if isinstance(shap_values, list) and len(shap_values) == 2:
                values = shap_values[1][0]  # Class 1 (High)
            else:
                values = shap_values[0]  # If only one output (regressor or binary fallback)
                if len(values.shape) > 1:
                    values = values[0]

            explanation = []
            for i, feat_name in enumerate(self.feature_names):
                val = values[i]
                explanation.append({
                    'feature': feat_name,
                    'importance': round(abs(val), 4),
                    'impact': 'positive' if val > 0 else 'negative',
                    'description': feature_metadata.get(feat_name, '')
                })

            explanation = sorted(explanation, key=lambda x: x['importance'], reverse=True)
            return explanation

        except Exception as e:
            print(f"⚠ SHAP explanation failed: {e}")
            return []

    
    def predict(self, data):
        """Make predictions using the loaded models"""
        if not self.model_loaded:
            raise Exception("Models not loaded properly")
        
        try:
            # Preprocess input
            input_features = self.preprocess_input(data)
            
            # Reshape for prediction
            input_features = input_features.reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                try:
                    scaled_features = self.scaler.transform(input_features)
                    print("✓ Features scaled using loaded scaler")
                except Exception as scale_error:
                    print(f"⚠ Warning: Scaler transform failed: {scale_error}")
                    print("Using unscaled features")
                    scaled_features = input_features
            else:
                print("⚠ No scaler available, using unscaled features")
                scaled_features = input_features
            
            # Make predictions
            mpi_hcr = self.regression_model.predict(scaled_features)[0]
            poverty_level = self.classification_model.predict(scaled_features)[0]
            
            # Get prediction probabilities for classification
            try:
                poverty_prob = self.classification_model.predict_proba(scaled_features)[0]
                prob_low = poverty_prob[0]
                prob_high = poverty_prob[1]
            except:
                prob_low = 0.5
                prob_high = 0.5
            
            explanation = self.explain_prediction(scaled_features)
            return {
                'mpiHcr': float(mpi_hcr),
                'povertyLevel': 'High' if poverty_level == 1 else 'Low',
                'confidence': {
                    'low_risk_probability': float(prob_low),
                    'high_risk_probability': float(prob_high)
                },
                'explanation': explanation
            }

            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise e

# Initialize the API
api = PovertyPredictionAPI()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        print(f"Received data: {data}")
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = api.predict(data)
        print(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model_loaded,
        'scaler_available': api.scaler is not None,
        'feature_count': len(api.feature_names)
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get the list of features expected by the model"""
    return jsonify({
        'features': api.feature_names,
        'count': len(api.feature_names)
    })


if __name__ == '__main__':
    # Load models at startup
    print("Starting Poverty Prediction API...")
    api.load_models()
    app.run(debug=True, port=5000)
