from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved models
disease_model = joblib.load('disease_model.pkl')
precaution_model = joblib.load('precaution_model.pkl')

# Define a simple function for prediction
def predict_disease_and_precaution(symptom_text):
    # Predict the disease based on symptoms
    disease = disease_model.predict([symptom_text])[0]

    # Generate the precaution input for the model
    precaution_input = f"{disease}: {symptom_text}"
    precaution = precaution_model.predict([precaution_input])[0]

    return disease, precaution

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# AJAX Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    symptoms = request.form['symptoms']

    # Validate input
    if not symptoms or len(symptoms.split(',')) < 3:
        return jsonify({'error': 'Please enter at least 3 symptoms separated by commas'})

    # Get predictions
    try:
        disease, precaution = predict_disease_and_precaution(symptoms)
        return jsonify({
            'disease': disease,
            'precaution': precaution
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)