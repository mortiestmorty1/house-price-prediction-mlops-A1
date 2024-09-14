import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Columns in the dataset:", data.columns)  # Check column names
    return data

# Preprocess data (convert lot size and handle missing values)
def preprocess_data(data):
    # Convert 'acre' to square feet
    def convert_lot_size(row):
        if row['lot_size_units'] == 'acre':  # Updated to 'lot_size_units'
            return row['lot_size'] * 43560
        return row['lot_size']

    # Apply conversion (assuming the column is named correctly)
    data['lot_size'] = data.apply(convert_lot_size, axis=1)
    
    # Fill missing lot_size values with the mean (or other imputation method)
    data['lot_size'].fillna(data['lot_size'].mean(), inplace=True)
    
    # Drop rows where essential columns are missing
    data.dropna(subset=['size', 'beds', 'baths'], inplace=True)
    
    # Inspect the result
    print(data.head())
    return data

# Select features and target
def select_features(data):
    X = data[['beds', 'baths', 'size', 'lot_size']]
    y = data['price']
    return X, y

# Split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Full preprocessing function
def full_preprocessing(file_path):
    data = load_data(file_path)
    data = preprocess_data(data)
    X, y = select_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X, y

# Example usage
X_train_scaled, X_test_scaled, y_train, y_test, X, y = full_preprocessing('data/train.csv')

# Save the preprocessed data
preprocessed_data = pd.concat([X, y], axis=1)
preprocessed_data.to_csv('data/preprocessed_train.csv', index=False)
print("Preprocessed data saved to data/preprocessed_train.csv")

print("Preprocessing complete!")
