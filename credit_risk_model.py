import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the credit risk dataset
data = pd.read_excel("default of credit card clients.xls", header=1)
# Display the first 5 rows
print(data.head())

# Display dataset size
print("\nDataset shape:", data.shape)

print("\nColumn names:")
print(data.columns.tolist())

#Target distribution
print("\nTarget distribution:")
print(data["default payment next month"].value_counts())

# Remove customer ID because it is not a useful prediction feature
data = data.drop("ID", axis=1)

print("\nDataset after removing ID:")
print(data.head())

print("\nNew dataset shape:", data.shape)

# Separate features and target
X = data.drop("default payment next month", axis=1)
y = data["default payment next month"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)

test_data = data.drop(train_data.index)

X_train = train_data.drop("default payment next month", axis=1)
y_train = train_data["default payment next month"]

X_test = test_data.drop("default payment next month", axis=1)
y_test = test_data["default payment next month"]

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the baseline model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

print("\nPredictions completed!")
print("First 10 predictions:", y_pred[:10])