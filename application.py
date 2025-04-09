from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load models
model = pickle.load(open("Model/modelforPrediction.pkl", "rb"))
scaler = pickle.load(open("Model/standardScaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/home.html')
def serve_home():
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Create numpy array with correct shape
        new_data = np.array([[Pregnancies, Glucose, BloodPressure, 
                            SkinThickness, Insulin, BMI, 
                            DiabetesPedigreeFunction, Age]])
        
        # Scale and predict
        new_data = scaler.transform(new_data)
        prediction = model.predict(new_data)

        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        return render_template('single_prediction.html', prediction_result=result)
    
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)