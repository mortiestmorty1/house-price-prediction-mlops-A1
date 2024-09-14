from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        beds = float(request.form['beds'])
        baths = float(request.form['baths'])
        size = float(request.form['size'])
        lot_size = float(request.form['lot_size'])

        # Make a prediction using the model
        features = np.array([[beds, baths, size, lot_size]])
        prediction = model.predict(features)

        # Return the result as JSON
        return jsonify({'price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
