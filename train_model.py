import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heart_disease_data.csv")

# Define features and target
X = df.drop(columns=["target"])
y = df["target"]

# Save feature names for later use
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model.pkl")
