import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X, y

# 2. Model Training
def train_model(X_train, y_train):
    # Initialize and train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# 3. Model Evaluation
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import classification_report, accuracy_score
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report and accuracy
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

# 4. Feature Importance
def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_imp = pd.DataFrame(sorted(zip(importances, feature_names), reverse=True), 
                               columns=['Importance', 'Feature'])
    return feature_imp

# Main Training and Saving Function
def train_and_save_model(dataset_path):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, X, y = load_and_preprocess_data(dataset_path)
    
    # Train the model on full dataset
    full_X_scaled = scaler.transform(X)
    model = train_model(full_X_scaled, y)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Get feature importance
    feature_importance = get_feature_importance(model, feature_names)
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model and scaler
    joblib.dump(model, 'models/crop_recommendation_model.joblib')
    joblib.dump(scaler, 'models/crop_recommendation_scaler.joblib')
    
    print("\nModel and Scaler saved successfully!")

# Run the training
if __name__ == '__main__':
    # Make sure you have the dataset in the current directory or provide full path
    train_and_save_model('datasets/crop_recommendation_dataset.csv')

# Prediction Function
def predict_crop(model, scaler, input_features):
    # Ensure input is in the correct format
    input_array = np.array(input_features).reshape(1, -1)
    
    # Scale the input
    scaled_input = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    return prediction[0]