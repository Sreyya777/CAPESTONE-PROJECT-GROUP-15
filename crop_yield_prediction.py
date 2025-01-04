import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_csv('Crop_Yield_Prediction.csv')

# Specify features and target variable
X = data.drop('Yield', axis=1)
y = data['Yield']

# Identify categorical and numerical features
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Get the unique list of crops for validation
valid_crops = X['Crop'].str.title().unique()

# Function to get user input and predict yield
def YieldPredictor():
    print("Please enter the following details to predict crop yield:")

    # Validate crop input
    crop = input("Crop (e.g., Rice, Wheat): ").strip().title()
    if crop not in valid_crops:
        print("Invalid crop name. Please enter a valid crop from the dataset.")
        return

    try:
        # Collect other inputs with error handling for invalid values
        nitrogen = float(input("Nitrogen (N): "))
        phosphorus = float(input("Phosphorus (P): "))
        potassium = float(input("Potassium (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        pH_value = float(input("pH Value: "))
        rainfall = float(input("Rainfall (mm): "))

        # Create a DataFrame for the new input
        new_input = pd.DataFrame({
            'Crop': [crop],
            'Nitrogen': [nitrogen],
            'Phosphorus': [phosphorus],
            'Potassium': [potassium],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH_Value': [pH_value],
            'Rainfall': [rainfall]
        })

        # Predict yield
        predicted_yield = model.predict(new_input)
        print(f'Predicted Yield for {crop}: {predicted_yield[0]}')

    except ValueError:
        print("Invalid input. Please enter numeric values for Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, and Rainfall.")

# Main loop
while True:
    print("\nWould you like to:")
    print("1. Predict crop yield")
    print("2. Exit")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        YieldPredictor()
    elif choice == '2':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")
