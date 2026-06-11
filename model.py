import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("diabetes.csv")

# Replace 0s with median for realistic medical values
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train multiple models and compare
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, pred) * 100, 2)
    results[name] = acc
    print(f"{name}: {acc}%")

# Pick best model
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest Model: {best_name} — {results[best_name]}%")

# Save
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(results, open("model_results.pkl", "wb"))
pickle.dump(best_name, open("best_model_name.pkl", "wb"))
print("All files saved!")