from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    hours = float(request.form['hours'])
    prediction = float(model.predict(np.array([[hours]]))[0])

    return render_template("index.html", 
                           prediction_text=f"Predicted Marks: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
