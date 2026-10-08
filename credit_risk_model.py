import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
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
model = LogisticRegression(max_iter=1000, class_weight="balanced")


model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Calculate ROC-AUC for the balanced model
y_prob = model.predict_proba(X_test_scaled)[:, 1]
balanced_auc = roc_auc_score(y_test, y_prob)

print("\nBalanced Model ROC-AUC:", balanced_auc)

# Train the original Logistic Regression model
original_model = LogisticRegression(max_iter=1000)

original_model.fit(X_train_scaled, y_train)

# Make predictions using the original model
original_y_pred = original_model.predict(X_test_scaled)

# Calculate ROC-AUC for the original model
original_y_prob = original_model.predict_proba(X_test_scaled)[:, 1]
original_auc = roc_auc_score(y_test, original_y_prob)

print("\nOriginal Model ROC-AUC:", original_auc)

# Calculate original model accuracy
original_accuracy = accuracy_score(y_test, original_y_pred)

print("\nOriginal Model Accuracy:", original_accuracy)

print("\nPredictions completed!")
print("First 10 predictions:", y_pred[:10])

#Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Balanced model metrics
balanced_report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

balanced_recall = balanced_report["1"]["recall"]
balanced_f1 = balanced_report["1"]["f1-score"]

print("\nBalanced Model Class 1 Recall:", balanced_recall)
print("Balanced Model Class 1 F1-score:", balanced_f1)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make predictions using Random Forest
rf_y_pred = rf_model.predict(X_test)

# Calculate Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_y_pred)

rf_report = classification_report(
    y_test,
    rf_y_pred,
    output_dict=True
)

rf_recall = rf_report["1"]["recall"]
rf_f1 = rf_report["1"]["f1-score"]

# Calculate ROC-AUC
rf_y_prob = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_y_prob)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Class 1 Recall:", rf_recall)
print("Random Forest Class 1 F1-score:", rf_f1)
print("Random Forest ROC-AUC:", rf_auc)

#Precision, Recall, F1-score

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)