import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load crop recommendation model and scaler
crop_rec_model = joblib.load('models/crop_recommendation_model.joblib')
crop_rec_scaler = joblib.load('models/crop_recommendation_scaler.joblib')

# Load crop yield prediction model
def load_yield_prediction_model():
    # Load your dataset
    data = pd.read_csv('datasets/Crop_Yield_Prediction.csv')

    # Specify features and target variable
    X = data.drop('Yield', axis=1)
    y = data['Yield']

    # Identify categorical and numerical features
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    categorical_features = ['Crop']
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Full model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X, y)
    return model

# Load yield prediction model
yield_prediction_model = load_yield_prediction_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    recommendation = None
    if request.method == 'POST':
        try:
            # Get input features
            input_features = [
                float(request.form['nitrogen']),
                float(request.form['phosphorus']),
                float(request.form['potassium']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            
            # Reshape input and scale
            input_array = np.array(input_features).reshape(1, -1)
            scaled_input = crop_rec_scaler.transform(input_array)
            
            # Predict crop
            recommendation = crop_rec_model.predict(scaled_input)[0]
        
        except ValueError:
            recommendation = "Please enter valid numerical inputs"
    
    return render_template('crop_recommendation.html', recommendation=recommendation)

@app.route('/crop-yield', methods=['GET', 'POST'])
def crop_yield():
    prediction = None
    valid_crops = None
    
    # Get unique crops from the dataset
    data = pd.read_csv('datasets/Crop_Yield_Prediction.csv')
    valid_crops = sorted(data['Crop'].unique())
    
    if request.method == 'POST':
        try:
            # Prepare input data
            new_input = pd.DataFrame({
                'Crop': [request.form['crop'].strip().title()],
                'Nitrogen': [float(request.form['nitrogen'])],
                'Phosphorus': [float(request.form['phosphorus'])],
                'Potassium': [float(request.form['potassium'])],
                'Temperature': [float(request.form['temperature'])],
                'Humidity': [float(request.form['humidity'])],
                'pH_Value': [float(request.form['ph'])],
                'Rainfall': [float(request.form['rainfall'])]
            })
            
            # Predict yield
            prediction = yield_prediction_model.predict(new_input)[0]
        
        except ValueError:
            prediction = "Please enter valid numerical inputs"
    
    return render_template('crop_yield.html', prediction=prediction, valid_crops=valid_crops)

if __name__ == '__main__':
    app.run(debug=True)