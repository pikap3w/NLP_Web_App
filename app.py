import joblib
from flask import Flask, render_template, request

# Initialize app
app = Flask(__name__, template_folder='./templates', static_folder='../static')

# Load pickled models
input_transformer = joblib.load(open('static/models/input_transformer.pkl', 'rb'))
model = joblib.load(open('static/models/review_sentiment.pkl', 'rb'))

# Global variables for data persistence across requests
model_input = ""
model_output = ""


# Main index page route
@app.route('/')
def home():
    return render_template('index.html', image_filename='img/happy.webp', display_mode='none')


# Route for predicting sentiment analysis and classifier
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve global variables to store input and output
    global model_input
    global model_output

    # Get text from the incoming request (submitted on predict button click)
    text = request.form['input_text']

    # Convert text to model input vector
    final_features = input_transformer.transform([text])

    # Use classifier's predict method to get prediction
    prediction = model.predict(final_features)

    # Store model input and output
    model_input = text
    model_output = prediction[0]

    return model_output


# Route for incremental training of model
@app.route('/save_pred', methods=['POST'])
def save_pred():
    return render_template('index.html', image_filename='img/happy.webp', display_mode='none')


if __name__ == '__main__':
    app.run(debug=True)
