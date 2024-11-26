from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model from the 'model' directory
model_path = os.path.join('model', 'random_forest_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html', prediction=None, aqi_value=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        try:
            so2 = float(request.form['so2'])
            no2 = float(request.form['no2'])
            rspm = float(request.form['rspm'])
            spm = float(request.form['spm'])
        except ValueError:
            return render_template('index.html', error="Please enter valid numeric values.", prediction=None, aqi_value=None)

        # Create input array for the model
        features = np.array([[so2, no2, rspm, spm]])

        # Make prediction
        prediction = model.predict(features)

        # Calculate AQI value (Assuming it is a simple average for this example)
        # You can change this calculation as per your AQI formula
        aqi_value = (so2 + no2 + rspm + spm) / 4

        return render_template('index.html', prediction=prediction[0], aqi_value=aqi_value)

if __name__ == '__main__':
    app.run(debug=True)

try:
    from flask import Flask, request, render_template
    print("Flask module imported successfully")
except Exception as e:
    print(f"Error importing Flask: {e}")

