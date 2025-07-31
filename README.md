# Poverty Prediction Backend

This project is a Flask application designed to predict poverty levels and the Multidimensional Poverty Index Headcount Ratio (MPI HCR) based on demographic data. It utilizes machine learning models for classification and regression tasks.

## Project Structure

```
poverty-prediction-backend
├── app.py                     # Main entry point of the Flask application
├── models                     # Directory containing trained models
│   ├── poverty_classification_model_v1.0_20250728_143203.pkl
│   ├── poverty_regression_model_v1.0_20250728_143203.pkl
│   └── scaler_v1.0_20250728_143203.pkl
├── requirements.txt           # List of dependencies for the project
├── utils                      # Utility functions for model handling
│   └── model_utils.py
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd poverty-prediction-backend
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application:**
   ```
   python app.py
   ```

2. **Access the API:**
   The application will be running on `http://127.0.0.1:5000`. You can send POST requests to the appropriate endpoints for predictions.

## API Endpoints

- **POST /predict**
  - Description: Predicts the poverty level and MPI HCR based on provided demographic data.
  - Request Body: JSON object containing demographic information.
  - Response: JSON object with prediction results.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
