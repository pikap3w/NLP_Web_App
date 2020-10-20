import csv

import joblib
from flask import Flask, render_template, request

# Initialize app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

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
    # Retrieve global variables
    global model_input
    global model_output

    # Vectorize user input
    final_features = input_transformer.transform([model_input])

    # Get user's button choice (correct/incorrect)
    save_type = request.form['save_type']

    # Return text
    return_text = 'The weights were strengthened. Thank you for teaching me!'

    # Modify global variable if user selected "incorrect"
    if save_type == 'incorrect':
        return_text = 'The weights were changed. Thank you for correcting me!'
        if model_output == 'positive':
            model_output = 'negative'
        elif model_output == 'negative':
            model_output = 'positive'
        else:
            print("Error: Model output was neither negative nor positive")

    # Strengthen weight of particular connection
    max_iter = 100
    counter = 0
    for i in range(0, max_iter):
        model.partial_fit(final_features, [model_output])
        if model.predict(final_features) == [model_output]:
            counter = i
            break

    # Save trained model pickle
    print(app.static_folder)
    joblib.dump(model, app.static_folder + '/models/review_sentiment.pkl')

    # Fields inside CSV to store for retrain verification
    fields = [model_input, model_output, counter]

    # Retrain model
    with open((app.root_path + '/user_teaching_data.csv'), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

    #  Return confirmation code for user
    return return_text


# TODO: Add validations for input text (cannot submit empty, etc.)
# TODO: Add prediction image and prediction text to html
# TODO: Add AJAX, jQuery for SPA

if __name__ == '__main__':
    app.run(debug=True)
