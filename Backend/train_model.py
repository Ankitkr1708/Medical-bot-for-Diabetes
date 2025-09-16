# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("Starting model training...")

# 1. Load the dataset
try:
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv' is in the same folder.")
    exit()

# 2. Prepare the data
# The first column 'Diabetes_binary' is our target, the rest are features
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# 4. Initialize and train the model
model = LogisticRegression(max_iter=1000)
print("Training the Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
joblib.dump(model, 'diabetes_model.pkl')
print("Model saved successfully as 'diabetes_model.pkl'")
