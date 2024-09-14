import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load preprocessed data directly from the CSV
def load_preprocessed_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['beds', 'baths', 'size', 'lot_size']]
    y = data['price']
    return X, y

# Model training functions
def train_model(X_train, y_train):
    model = LinearRegression()  # Using Linear Regression model
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model RMSE: {rmse}")
    return rmse

# Save the trained model
def save_model(model, file_name='model.pkl'):
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")

# Full workflow
def full_workflow(file_path):
    # Load the preprocessed data
    X, y = load_preprocessed_data(file_path)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model)

# Run the workflow using the preprocessed data
if __name__ == '__main__':
    full_workflow('data/preprocessed_train.csv')
