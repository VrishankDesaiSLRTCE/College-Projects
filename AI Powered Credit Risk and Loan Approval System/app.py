from flask import Flask, render_template, request
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ← Put it here, right after creating 'app'
app.config['PROPAGATE_EXCEPTIONS'] = True

model = joblib.load('credit_model.pkl')

def init_db():
    conn = sqlite3.connect('applications.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY, name TEXT, email TEXT, age INTEGER, income INTEGER,
        job_stability INTEGER, loan_amount INTEGER, credit_history INTEGER,
        risk_prob REAL, decision TEXT, explanation TEXT,
        suggested_loan INTEGER, emi REAL, tenure INTEGER, timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    age = int(request.form['age'])
    income = int(request.form['income'])
    job_stability = int(request.form['job_stability'])
    loan_amount = int(request.form['loan_amount'])
    credit_history = int(request.form['credit_history'])

    input_data = pd.DataFrame([[age, income, job_stability, loan_amount, credit_history]],
                              columns=['age', 'income', 'job_stability', 'loan_amount', 'credit_history'])

    prediction = model.predict(input_data)[0]
    risk_prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1 or risk_prob > 45:
        decision = "Rejected"
        if credit_history == 1:
            explanation = "Bad credit history"
        elif income < loan_amount * 2:
            explanation = "Income too low compared to loan"
        elif job_stability < 3:
            explanation = "Very low job stability"
        else:
            explanation = f"High risk ({risk_prob:.1f}%)"
        suggested_loan = 0
        emi = 0
        tenure = 0
    else:
        decision = "Approved"
        explanation = f"Low risk profile ({risk_prob:.1f}%)"
        suggested_loan = min(loan_amount, income * 5)
        tenure = 36
        monthly_rate = 0.09 / 12
        emi = (suggested_loan * monthly_rate * (1 + monthly_rate)**tenure) / ((1 + monthly_rate)**tenure - 1)

    # Save to database
    conn = sqlite3.connect('applications.db')
    c = conn.cursor()
    c.execute("INSERT INTO applications VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (name, email, age, income, job_stability, loan_amount, credit_history,
               risk_prob, decision, explanation, suggested_loan, emi, tenure, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

    return render_template('result.html', name=name, decision=decision, explanation=explanation,
                           risk_prob=round(risk_prob,1), suggested_loan=suggested_loan,
                           emi=round(emi,2), tenure=tenure)

@app.route('/history')
def history():
    conn = sqlite3.connect('applications.db')
    c = conn.cursor()
    c.execute("SELECT * FROM applications ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', applications=rows)

if __name__ == '__main__':
    app.run(debug=True)