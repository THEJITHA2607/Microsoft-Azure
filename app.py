from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    previous_score = float(request.form['previous_score'])

    prediction = model.predict([[study_hours, attendance, previous_score]])

    result = "PASS" if prediction[0] == 1 else "FAIL"

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
