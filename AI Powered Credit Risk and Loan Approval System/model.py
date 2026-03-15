import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('credit_data.csv')

features = data[['age', 'income', 'job_stability', 'loan_amount', 'credit_history']]
target = data['risk']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'credit_model.pkl')
print("Model trained successfully! Accuracy:", model.score(X_test, y_test)*100, "%")